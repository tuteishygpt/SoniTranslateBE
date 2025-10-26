import os
import re
import gc
import json
import glob
import traceback
import logging
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    class _TqdmFallback:
        """Minimal stand-in for ``tqdm`` when the dependency is unavailable."""

        def __init__(self, iterable=None, total=None, desc=None, **kwargs):
            self._iterable = iterable
            self.total = total
            self.desc = desc

        def __iter__(self):
            if self._iterable is None:
                return iter(range(self.total or 0))
            return iter(self._iterable)

        def update(self, n=1):
            return None

        def close(self):
            return None

    def tqdm(iterable=None, *args, **kwargs):  # type: ignore
        return _TqdmFallback(iterable=iterable, **kwargs)

try:
    import librosa  # type: ignore
except ImportError:  # pragma: no cover
    librosa = None  # type: ignore

import numpy as np
import soundfile as sf

try:
    import torch
except ImportError:
    torch = None  # type: ignore

# мы будзем адкрыта карыстацца нізкаўзроўневым XTTS
# вельмі важна: у цябе код мадэлі ляжыць у xtts_local/TTS/...
# таму мы імпартуем адтуль, а не з pypiшнай TTS
from xtts_local.TTS.tts.configs.xtts_config import XttsConfig
from xtts_local.TTS.tts.models.xtts import Xtts

from .language_configuration import fix_code_language
from .utils import (
    download_manager,
    create_directories,
    copy_files,
    rename_file,
    remove_directory_contents,
    remove_files,
    run_command,
    write_chunked,
)
from .logging_setup import logger


class TTS_OperationError(Exception):
    def __init__(self, message="The operation did not complete successfully."):
        self.message = message
        super().__init__(self.message)


def verify_saved_file_and_size(filename):
    if not os.path.exists(filename):
        raise TTS_OperationError(f"File '{filename}' was not saved.")
    if os.path.getsize(filename) == 0:
        raise TTS_OperationError(
            f"File '{filename}' has a zero size. "
            "Related to incorrect TTS for the target language"
        )


def error_handling_in_tts(error, segment, TRANSLATE_AUDIO_TO, filename):
    traceback.print_exc()
    logger.error(f"Error: {str(error)}")
    try:
        sample_rate_aux = 22050
        duration = float(segment["end"]) - float(segment["start"])
        data = np.zeros(int(sample_rate_aux * duration)).astype(np.float32)
        write_chunked(
            filename, data, sample_rate_aux, format="ogg", subtype="vorbis"
        )

        logger.warning(
            "Primary TTS failed. Silent audio will be used instead."
        )
        verify_saved_file_and_size(filename)
    except Exception as err:
        logger.critical(f"Error creating fallback audio: {str(err)}")
        raise


def pad_array(array, sr):
    # абразаць лішнюю цішыню (як раней)
    if isinstance(array, list):
        array = np.array(array)

    if not array.shape[0]:
        raise ValueError("The generated audio does not contain any data")

    valid_indices = np.where(np.abs(array) > 0.001)[0]

    if len(valid_indices) == 0:
        logger.debug(f"No valid indices: {array}")
        return array

    try:
        pad_indice = int(0.1 * sr)
        start_pad = max(0, valid_indices[0] - pad_indice)
        end_pad = min(len(array), valid_indices[-1] + 1 + pad_indice)
        padded_array = array[start_pad:end_pad]
        return padded_array
    except Exception as error:
        logger.error(str(error))
        return array


def get_audio_duration(filename):
    if librosa is not None:
        return librosa.get_duration(filename=filename)

    info = sf.info(filename)
    if not info.samplerate:
        raise TTS_OperationError(
            f"Unable to determine duration for '{filename}' without a samplerate"
        )
    return info.frames / info.samplerate


def edge_tts_voices_list():
    """stub for backward compat"""
    logger.debug("edge_tts_voices_list called but Edge TTS backend is disabled")
    return []


def piper_tts_voices_list():
    """stub for backward compat"""
    logger.debug("piper_tts_voices_list called but Piper TTS backend is disabled")
    return []


# =====================================
# Coqui XTTS custom model management
# =====================================

CUSTOM_COQUI_MODELS = {
    "be": {
        "repo": "archivartaunik/BE_XTTS_V2_10ep250k",
        "files": {
            "model.pth": "model.pth",
            "config.json": "config.json",
            "vocab.json": "vocab.json",
            "dvae.pth": "dvae.pth",
            "mel_stats.pth": "mel_stats.pth",
            "speakers_xtts.pth": "speakers_xtts.pth",
        },
    }
}


def _safe_join(*parts):
    return os.path.join(*parts)


def _sanitize_infinity(obj):
    # каб json.dump не ламаўся на float('inf')
    if isinstance(obj, dict):
        return {k: _sanitize_infinity(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_infinity(v) for v in obj]
    if isinstance(obj, float):
        if obj == float("inf"):
            return 9_999_999_999.0
    return obj


def _patch_coqui_config(orig_config_path: str, model_dir: str) -> str:
    """
    Некаторыя fine-tune канфігі XTTS маюць absolute-шляхі і спасылкі кшталту
    /checkpoints/... . Мы будуем config.runtime.json
    са шляхамі ўнутры model_dir.
    """
    runtime_config_path = _safe_join(model_dir, "config.runtime.json")

    try:
        with open(orig_config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        logger.error(f"Cannot load custom XTTS config: {e}")
        return orig_config_path  # fallback

    model_args = cfg.get("model_args") or cfg.get("model", {}).get("args")
    if model_args and isinstance(model_args, dict):

        def _rp(local_name: str) -> str:
            return _safe_join(model_dir, local_name)

        # tokenizer / vocab
        if "tokenizer_file" in model_args:
            model_args["tokenizer_file"] = _rp("vocab.json")

        # mel stats
        if "mel_norm_file" in model_args:
            model_args["mel_norm_file"] = _rp("mel_stats.pth")

        # dvae
        if "dvae_checkpoint" in model_args:
            model_args["dvae_checkpoint"] = _rp("dvae.pth")

        # main xtts checkpoint
        if "xtts_checkpoint" in model_args:
            model_args["xtts_checkpoint"] = _rp("model.pth")

    if "model_dir" in cfg:
        cfg["model_dir"] = model_dir

    cfg = _sanitize_infinity(cfg)

    try:
        with open(runtime_config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        logger.info(f"Patched XTTS config written to {runtime_config_path}")
        return runtime_config_path
    except Exception as e:
        logger.error(f"Cannot write runtime XTTS config: {e}")
        return orig_config_path


def download_custom_coqui_model(language_code):
    """
    - сцягваем усе патрэбныя файлы з HuggingFace
    - робім патчаны config.runtime.json з лакальнымі шляхамі
    - вяртаем шляхі, якія патрэбны для ініцыялізацыі мадэлі
    """
    model_meta = CUSTOM_COQUI_MODELS.get(language_code)
    if not model_meta:
        return {}

    repo = model_meta["repo"]
    sanitized_repo = repo.replace("/", "_")
    model_dir = _safe_join("COQUI_MODELS", sanitized_repo)
    create_directories(model_dir)

    downloaded_paths = {}
    for local_name, remote_name in model_meta["files"].items():
        url = f"https://huggingface.co/{repo}/resolve/main/{remote_name}"
        try:
            downloaded_file = download_manager(url=url, path=model_dir, progress=False)
        except Exception as error:
            logger.error(
                "Unable to download '%s' for custom Coqui model '%s': %s",
                remote_name,
                repo,
                error,
            )
            return {}

        if not downloaded_file or not Path(downloaded_file).is_file():
            logger.error(
                "Custom Coqui model file missing after download: %s", remote_name
            )
            return {}

        downloaded_paths[local_name] = downloaded_file

    runtime_config_path = _patch_coqui_config(
        downloaded_paths["config.json"], model_dir
    )

    final_paths = {
        "checkpoint_dir": model_dir,  # тэчка з усім
        "model_path": _safe_join(model_dir, "model.pth"),  # сам чэкпойнт
        "config_path": runtime_config_path,               # патчаны config
        "vocab_path": _safe_join(model_dir, "vocab.json"),
    }

    return final_paths


def coqui_xtts_voices_list():
    """
    вяртаем спіс даступных reference узораў галасоў у _XTTS_
    (прапануем карыстальніку)
    """
    main_folder = "_XTTS_"
    pattern_coqui = re.compile(r".+\.(wav|mp3|ogg|m4a)$")
    pattern_automatic_speaker = re.compile(r"AUTOMATIC_SPEAKER_\d+\.wav$")

    if not os.path.isdir(main_folder):
        return ["_XTTS_/AUTOMATIC.wav"]

    wav_voices = [
        "_XTTS_/" + f
        for f in os.listdir(main_folder)
        if os.path.isfile(os.path.join(main_folder, f))
        and pattern_coqui.match(f)
        and not pattern_automatic_speaker.match(f)
    ]

    return ["_XTTS_/AUTOMATIC.wav"] + wav_voices


def seconds_to_hhmmss_ms(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return "%02d:%02d:%02d.%03d" % (hours, minutes, int(seconds), milliseconds)


def audio_trimming(audio_path, destination, start, end):
    if isinstance(start, (int, float)):
        start = seconds_to_hhmmss_ms(start)
    if isinstance(end, (int, float)):
        end = seconds_to_hhmmss_ms(end)

    file_directory = destination if destination else os.path.dirname(audio_path)

    file_name = os.path.splitext(os.path.basename(audio_path))[0]
    file_ = f"{file_name}_trim.wav"
    output_path = os.path.join(file_directory, file_)

    command = (
        f'ffmpeg -y -loglevel error -i "{audio_path}" '
        f"-ss {start} -to {end} "
        f'-acodec pcm_s16le -f wav "{output_path}"'
    )
    run_command(command)

    return output_path


def convert_to_xtts_good_sample(audio_path: str = "", destination: str = ""):
    file_directory = destination if destination else os.path.dirname(audio_path)

    file_name = os.path.splitext(os.path.basename(audio_path))[0]
    file_ = f"{file_name}_good_sample.wav"
    mono_path = os.path.join(file_directory, file_)

    # reference speaker must be mono 22.05kHz 16-bit
    command = (
        f'ffmpeg -y -loglevel error -i "{audio_path}" '
        f"-ac 1 -ar 22050 -sample_fmt s16 -f wav \"{mono_path}\""
    )
    run_command(command)

    return mono_path


def sanitize_file_name(file_name):
    import unicodedata
    normalized_name = unicodedata.normalize("NFKD", file_name)
    sanitized_name = re.sub(r"[^\w\s.-]", "_", normalized_name)
    return sanitized_name


def create_wav_file_vc(
    sample_name="",
    audio_wav="",
    start=None,
    end=None,
    output_final_path="_XTTS_",
    get_vocals_dereverb=True,
):
    sample_name = sample_name if sample_name else "default_name"
    sample_name = sanitize_file_name(sample_name)
    audio_wav = audio_wav if isinstance(audio_wav, str) else audio_wav.name

    BASE_DIR = "."
    output_dir = os.path.join(BASE_DIR, "clean_song_output")

    if start or end:
        audio_segment = audio_trimming(audio_wav, output_dir, start, end)
    else:
        audio_segment = audio_wav

    # выдаліць музыку / рэверберацыю з узорнага кавалка
    from .mdx_net import process_uvr_task
    try:
        _, _, _, _, audio_segment = process_uvr_task(
            orig_song_path=audio_segment,
            main_vocals=True,
            dereverb=get_vocals_dereverb,
        )
    except Exception as error:
        logger.error(str(error))

    sample = convert_to_xtts_good_sample(audio_segment)

    sample_name = f"{sample_name}.wav"
    sample_rename = rename_file(sample, sample_name)

    copy_files(sample_rename, output_final_path)

    final_sample = os.path.join(output_final_path, sample_name)
    if os.path.exists(final_sample):
        logger.info(final_sample)
        return final_sample
    else:
        raise Exception(f"Error wav: {final_sample}")


def create_new_files_for_vc(
    speakers_coqui,
    segments_base,
    dereverb_automatic=True
):
    # чысцім часовыя вынікі
    output_dir = os.path.join(".", "clean_song_output")
    remove_directory_contents(output_dir)

    for speaker in speakers_coqui:
        filtered_speaker = [
            segment
            for segment in segments_base
            if segment["speaker"] == speaker
        ]

        if len(filtered_speaker) > 4:
            filtered_speaker = filtered_speaker[1:]

        if filtered_speaker[0]["tts_name"] == "_XTTS_/AUTOMATIC.wav":
            name_automatic_wav = f"AUTOMATIC_{speaker}"
            automatic_path = f"_XTTS_/{name_automatic_wav}.wav"
            if os.path.exists(automatic_path):
                logger.info(f"WAV automatic {speaker} exists")
            else:
                wav_ok = False
                for seg in filtered_speaker:
                    duration = float(seg["end"]) - float(seg["start"])
                    if 7.0 < duration < 12.0:
                        logger.info(
                            f'Processing segment: {seg["start"]}, {seg["end"]}, '
                            f'{seg["speaker"]}, {duration}, {seg["text"]}'
                        )
                        create_wav_file_vc(
                            sample_name=name_automatic_wav,
                            audio_wav="audio.wav",
                            start=(float(seg["start"]) + 1.0),
                            end=(float(seg["end"]) - 1.0),
                            get_vocals_dereverb=dereverb_automatic,
                        )
                        wav_ok = True
                        break

                if not wav_ok:
                    # fallback: любы кавалак
                    logger.info("Taking the first segment for AUTOMATIC ref")
                    seg = filtered_speaker[0]
                    duration_full = float(seg["end"]) - float(seg["start"])
                    duration_full = max(2.0, min(duration_full, 9.0))

                    create_wav_file_vc(
                        sample_name=name_automatic_wav,
                        audio_wav="audio.wav",
                        start=(float(seg["start"])),
                        end=(float(seg["start"]) + duration_full),
                        get_vocals_dereverb=dereverb_automatic,
                    )


# ---------------------------
# helpers for the new XTTS inference path
# ---------------------------

def _split_sentences(text: str):
    """
    мы імкнёмся зрабіць падобна да прыкладу:
        from underthesea import sent_tokenize
    але underthesea працуе для в'етнамскай.
    тут: пробуем імпартаваць, калі няма – робім просты split па знаках канца сказа.
    """
    try:
        from underthesea import sent_tokenize  # type: ignore
        sents = sent_tokenize(text)
        # падстрахуемся што там не пустаты
        sents = [s.strip() for s in sents if s.strip()]
        if sents:
            return sents
    except Exception:
        pass

    # fallback: грубы падзел па .?! і захоўваем знакі
    # гэта не ідэальна для беларускай, але працуе лепш чым нічога
    chunks = re.split(r'([.?!…]+)', text)
    merged = []
    buf = ""
    for part in chunks:
        if re.match(r'[.?!…]+', part):
            buf += part
            merged.append(buf.strip())
            buf = ""
        else:
            if buf.strip():
                buf += " " + part
            else:
                buf = part
    if buf.strip():
        merged.append(buf.strip())

    merged = [m for m in merged if m]
    return merged or [text]


def _prepare_xtts_model(custom_model_paths, device):
    """
    Ствараем і вяртаем загружаную мадэль XTTS (Xtts)
    + гатовы config (XttsConfig) на аснове кастомнага fine-tune.
    """
    xtts_checkpoint = custom_model_paths["model_path"]
    xtts_config = custom_model_paths["config_path"]
    xtts_vocab = custom_model_paths["vocab_path"]

    # загружаем канфіг
    config = XttsConfig()
    config.load_json(xtts_config)

    # ініцыялізуем мадэль з канфіга
    xtts_model = Xtts.init_from_config(config)

    # грузім вагі
    # важна: fork XTTS чакае load_checkpoint(config, checkpoint_path, vocab_path, use_deepspeed=False)
    xtts_model.load_checkpoint(
        config,
        checkpoint_path=xtts_checkpoint,
        vocab_path=xtts_vocab,
        use_deepspeed=False,
    )

    # на GPU ці CPU
    xtts_model.to(device)

    logger.info("XTTS model loaded successfully!")
    return xtts_model, config


def _get_conditioning_for_ref(xtts_model, config, ref_path, cache_dict):
    """
    ref_path -> вяртаем (gpt_cond_latent, speaker_embedding)
    з кэшаваннем, каб не лічыць зноў для таго ж самага спектра.
    """
    if ref_path in cache_dict:
        return cache_dict[ref_path]

    gpt_cond_latent, speaker_embedding = xtts_model.get_conditioning_latents(
        audio_path=ref_path,
        gpt_cond_len=config.gpt_cond_len,
        max_ref_length=config.max_ref_len,
        sound_norm_refs=config.sound_norm_refs,
    )
    cache_dict[ref_path] = (gpt_cond_latent, speaker_embedding)
    return cache_dict[ref_path]


def _synthesize_sentence_chunks(
    xtts_model,
    config,
    text: str,
    lang: str,
    gpt_cond_latent,
    speaker_embedding,
):
    """
    як у прыкладзе: праганяем усе сказы асобна і клеім у адзін тэнзар.
    вяртаем np.float32 waveform (1D)
    """
    sents = _split_sentences(text)

    wav_chunks = []
    for sent in tqdm(sents):
        sent = sent.strip()
        if not sent:
            continue

        # параметры inference ўзятыя з прыкладу
        wav_chunk = xtts_model.inference(
            text=sent,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=0.1,
            length_penalty=1.0,
            repetition_penalty=10.0,
            top_k=10,
            top_p=0.3,
        )
        # inference вяртае dict са "wav"
        wav_tensor = torch.tensor(wav_chunk["wav"], dtype=torch.float32)

        wav_chunks.append(wav_tensor)

    if not wav_chunks:
        return np.zeros((0,), dtype=np.float32)

    out_wav = torch.cat(wav_chunks, dim=0).cpu().numpy().astype(np.float32)
    return out_wav


# ---------------------------
# ГАЛОЎНАЕ: segments_coqui_tts цяпер робіць усё "ўручную" праз Xtts
# ---------------------------

def segments_coqui_tts(
    filtered_coqui_segments,
    TRANSLATE_AUDIO_TO,
    model_id_coqui="UNUSED_WITH_DIRECT_XTTS",
    speakers_coqui=None,
    delete_previous_automatic=True,
    dereverb_automatic=True,
    emotion=None,
):
    """
    Новая версія, якая капіруе патэрн з твайго прыкладу:
    - не карыстаемся TTS.api
    - наўпрост грузім XttsConfig / Xtts
    - атрымліваем conditioning_latents ад reference WAV
    - робім inference() для кожнага сегмента
    - захоўваем у audio/{start}.ogg

    TRANSLATE_AUDIO_TO -> код мовы ("be", "en", ...)
    speakers_coqui -> спікеры, для якіх трэба генераваць
    """

    if torch is None:
        raise RuntimeError("PyTorch is required for XTTS inference.")

    # прывесці код мовы да фармату, які чакае XTTS (той жа, што мы перадалі раней)
    TRANSLATE_AUDIO_TO = fix_code_language(TRANSLATE_AUDIO_TO, syntax="coqui")

    supported_lang_coqui = [
        "zh-cn", "en", "fr", "de", "it", "pt", "pl", "tr", "ru",
        "nl", "cs", "ar", "es", "hu", "ko", "ja", "be",
    ]
    if TRANSLATE_AUDIO_TO not in supported_lang_coqui:
        raise TTS_OperationError(
            f"'{TRANSLATE_AUDIO_TO}' is not a supported language for XTTS"
        )

    # зачысціць AUTOMATIC_* калі патрэбна (мы працягваем гэтую паводзіны)
    if delete_previous_automatic and speakers_coqui:
        for spk in speakers_coqui:
            remove_files(f"_XTTS_/AUTOMATIC_{spk}.wav")

    # падрыхтаваць reference спікерскія файлы (AUTOMATIC_xx.wav і г.д.)
    directory_audios_vc = "_XTTS_"
    create_directories(directory_audios_vc)

    create_new_files_for_vc(
        speakers_coqui,
        filtered_coqui_segments["segments"],
        dereverb_automatic,
    )

    # выбіраем прыладу
    device_env = os.environ.get("SONITR_DEVICE")
    if device_env:
        device = device_env
    else:
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"

    # сцягваем і рыхтуем кастомную мадэль (напрыклад "be")
    custom_model_paths = download_custom_coqui_model(TRANSLATE_AUDIO_TO)
    if not custom_model_paths:
        # мы разлічваем на кастомную мадэль. калі няма - гэта крытычна
        raise RuntimeError(
            f"No custom XTTS model files for lang {TRANSLATE_AUDIO_TO}"
        )

    xtts_model, xtts_config = _prepare_xtts_model(custom_model_paths, device)

    # conditioning cache: ref_path -> (gpt_cond_latent, speaker_embedding)
    conditioning_cache = {}

    sampling_rate = 24000  # XTTS inference output SR

    # ідзём па сегментах, генерарам аўдыё
    for segment in tqdm(filtered_coqui_segments["segments"]):
        speaker = segment["speaker"]
        text = segment["text"]
        start = segment["start"]
        ref_voice_path = segment["tts_name"]
        if ref_voice_path == "_XTTS_/AUTOMATIC.wav":
            ref_voice_path = f"_XTTS_/AUTOMATIC_{speaker}.wav"

        filename = f"audio/{start}.ogg"
        logger.info(f"{text} >> {filename} (speaker {speaker}, ref {ref_voice_path})")

        try:
            # 1) дастаём latent голасу (кэшуем)
            gpt_cond_latent, speaker_embedding = _get_conditioning_for_ref(
                xtts_model,
                xtts_config,
                ref_voice_path,
                conditioning_cache,
            )

            # 2) запускаем inference з падзелам на сказы і канкатэнім
            wav_np = _synthesize_sentence_chunks(
                xtts_model,
                xtts_config,
                text,
                TRANSLATE_AUDIO_TO,
                gpt_cond_latent,
                speaker_embedding,
            )

            if wav_np.size == 0:
                raise RuntimeError("XTTS produced empty audio.")

            data_tts = pad_array(wav_np, sampling_rate)

            write_chunked(
                file=filename,
                samplerate=sampling_rate,
                data=data_tts,
                format="ogg",
                subtype="vorbis",
            )
            verify_saved_file_and_size(filename)

        except Exception as error:
            error_handling_in_tts(error, segment, TRANSLATE_AUDIO_TO, filename)

        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    # прыбраўшы мадэль з памяці
    try:
        del xtts_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as error:
        logger.error(str(error))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =====================================
# далейшы пайплайн (паскарэнне, voice conversion і г.д.)
# =====================================

def find_spkr(pattern, speaker_to_voice, segments):
    return [
        speaker
        for speaker, voice in speaker_to_voice.items()
        if pattern.match(voice) and any(
            segment["speaker"] == speaker for segment in segments
        )
    ]


def filter_by_speaker(speakers, segments):
    return {
        "segments": [
            segment
            for segment in segments
            if segment["speaker"] in speakers
        ]
    }


def audio_segmentation_to_voice(
    result_diarize,
    TRANSLATE_AUDIO_TO,
    is_gui,
    tts_voice00,
    tts_voice01="",
    tts_voice02="",
    tts_voice03="",
    tts_voice04="",
    tts_voice05="",
    tts_voice06="",
    tts_voice07="",
    tts_voice08="",
    tts_voice09="",
    tts_voice10="",
    tts_voice11="",
    dereverb_automatic=True,
    model_id_coqui="UNUSED_WITH_DIRECT_XTTS",
    delete_previous_automatic=True,
):
    """
    1. прыпісваем speaker і tts_name кожнаму сегменту
    2. вызначаем якія спікеры выкарыстоўваюць voice-clone (файл .wav / .mp3 / .ogg / .m4a)
    3. запускаем segments_coqui_tts(), які цяпер робіць прамое XTTS inference
    """

    remove_directory_contents("audio")

    speaker_to_voice = {
        "SPEAKER_00": tts_voice00,
        "SPEAKER_01": tts_voice01,
        "SPEAKER_02": tts_voice02,
        "SPEAKER_03": tts_voice03,
        "SPEAKER_04": tts_voice04,
        "SPEAKER_05": tts_voice05,
        "SPEAKER_06": tts_voice06,
        "SPEAKER_07": tts_voice07,
        "SPEAKER_08": tts_voice08,
        "SPEAKER_09": tts_voice09,
        "SPEAKER_10": tts_voice10,
        "SPEAKER_11": tts_voice11,
    }

    for segment in result_diarize["segments"]:
        if "speaker" not in segment:
            segment["speaker"] = "SPEAKER_00"
            logger.warning(
                "NO SPEAKER DETECT IN SEGMENT: First TTS will be used in the "
                f"segment time {segment['start'], segment['text']}"
            )
        segment["tts_name"] = speaker_to_voice[segment["speaker"]]

    pattern_coqui = re.compile(r".+\.(wav|mp3|ogg|m4a)$")

    all_segments = result_diarize["segments"]
    speakers_coqui = find_spkr(pattern_coqui, speaker_to_voice, all_segments)

    filtered_coqui = filter_by_speaker(speakers_coqui, all_segments)

    if filtered_coqui["segments"]:
        logger.info(f"XTTS inference for speakers: {speakers_coqui}")
        segments_coqui_tts(
            filtered_coqui,
            TRANSLATE_AUDIO_TO,
            model_id_coqui,
            speakers_coqui,
            delete_previous_automatic,
            dereverb_automatic,
        )

    # прыбіраем тэхнічнае поле
    [result.pop("tts_name", None) for result in result_diarize["segments"]]

    return speakers_coqui


def accelerate_segments(
    result_diarize,
    max_accelerate_audio,
    valid_speakers,
    acceleration_rate_regulation=False,
    folder_output="audio2",
):
    logger.info("Apply acceleration (time-stretch / atempo)")

    create_directories(f"{folder_output}/audio/")
    remove_directory_contents(f"{folder_output}/audio/")

    audio_files = []
    speakers_list = []

    max_count_segments_idx = len(result_diarize["segments"]) - 1

    for i, segment in tqdm(enumerate(result_diarize["segments"])):
        text = segment["text"]
        start = segment["start"]
        end = segment["end"]
        speaker = segment["speaker"]

        filename = f"audio/{start}.ogg"

        duration_true = end - start
        duration_tts = get_audio_duration(filename)

        acc_percentage = duration_tts / duration_true

        if acceleration_rate_regulation and acc_percentage >= 1.3:
            try:
                next_segment = result_diarize["segments"][
                    min(max_count_segments_idx, i + 1)
                ]
                next_start = next_segment["start"]
                next_speaker = next_segment["speaker"]
                duration_with_next_start = next_start - start

                if duration_with_next_start > duration_true:
                    extra_time = duration_with_next_start - duration_true
                    if speaker == next_speaker:
                        smoth_duration = duration_true + (extra_time * 0.5)
                    else:
                        smoth_duration = duration_true + (extra_time * 0.7)

                    acc_percentage = max(1.2, (duration_tts / smoth_duration))

            except Exception as error:
                logger.error(str(error))

        if acc_percentage > max_accelerate_audio:
            acc_percentage = max_accelerate_audio
        elif 0.8 <= acc_percentage <= 1.15:
            acc_percentage = 1.0
        elif acc_percentage < 0.8:
            acc_percentage = 0.8

        acc_percentage = round(acc_percentage + 0.0, 1)

        info_enc = "OGG"

        if acc_percentage == 1.0 and info_enc == "OGG":
            copy_files(filename, f"{folder_output}{os.sep}audio")
        else:
            os.system(
                f"ffmpeg -y -loglevel panic -i {filename} "
                f"-filter:a atempo={acc_percentage} {folder_output}/{filename}"
            )

        if logger.isEnabledFor(logging.DEBUG):
            duration_create = get_audio_duration(
                filename=f"{folder_output}/{filename}"
            )
            logger.debug(
                f"acc_percen is {acc_percentage}, tts duration "
                f"is {duration_tts}, new duration is {duration_create}"
                f", for {filename}"
            )

        audio_files.append(f"{folder_output}/{filename}")
        speaker_human = "TTS Speaker {:02d}".format(int(speaker[-2:]) + 1)
        speakers_list.append(speaker_human)

    return audio_files, speakers_list


# =====================================
# Voice conversion / tone color (OpenVoice / FreeVC)
# =====================================

def se_process_audio_segments(
    source_seg, tone_color_converter, device, remove_previous_processed=True
):
    source_audio_segs = glob.glob(f"{source_seg}/*.wav")
    if not source_audio_segs:
        raise ValueError(
            f"No audio segments found in {str(source_audio_segs)}"
        )

    source_se_path = os.path.join(source_seg, "se.pth")

    if os.path.isfile(source_se_path):
        se = torch.load(source_se_path).to(device)
        logger.debug(f"Previous created {source_se_path}")
    else:
        se = tone_color_converter.extract_se(source_audio_segs, source_se_path)

    return se


def create_wav_vc(
    valid_speakers,
    segments_base,
    audio_name,
    max_segments=10,
    target_dir="processed",
    get_vocals_dereverb=False,
):
    output_dir = os.path.join(".", target_dir)

    path_source_segments = []
    path_target_segments = []
    for speaker in valid_speakers:
        filtered_speaker = [
            segment
            for segment in segments_base
            if segment["speaker"] == speaker
        ]
        if len(filtered_speaker) > 4:
            filtered_speaker = filtered_speaker[1:]

        dir_name_speaker = speaker + audio_name
        dir_name_speaker_tts = "tts" + speaker + audio_name
        dir_path_speaker = os.path.join(output_dir, dir_name_speaker)
        dir_path_speaker_tts = os.path.join(output_dir, dir_name_speaker_tts)
        create_directories([dir_path_speaker, dir_path_speaker_tts])

        path_target_segments.append(dir_path_speaker)
        path_source_segments.append(dir_path_speaker_tts)

        max_segments_count = 0
        for seg in filtered_speaker:
            duration = float(seg["end"]) - float(seg["start"])
            if 3.0 < duration < 18.0:
                logger.info(
                    f'Processing segment: {seg["start"]}, {seg["end"]}, '
                    f'{seg["speaker"]}, {duration}, {seg["text"]}'
                )
                name_new_wav = str(seg["start"])

                check_segment_audio_target_file = os.path.join(
                    dir_path_speaker, f"{name_new_wav}.wav"
                )

                if os.path.exists(check_segment_audio_target_file):
                    logger.debug(
                        "Segment vc source exists: "
                        f"{check_segment_audio_target_file}"
                    )
                else:
                    create_wav_file_vc(
                        sample_name=name_new_wav,
                        audio_wav="audio.wav",
                        start=(float(seg["start"]) + 1.0),
                        end=(float(seg["end"]) - 1.0),
                        output_final_path=dir_path_speaker,
                        get_vocals_dereverb=get_vocals_dereverb,
                    )

                    file_name_tts = f"audio2/audio/{str(seg['start'])}.ogg"
                    convert_to_xtts_good_sample(
                        file_name_tts, dir_path_speaker_tts
                    )

                max_segments_count += 1
                if max_segments_count == max_segments:
                    break

        if max_segments_count == 0:
            logger.info("Taking the first segment (fallback)")
            seg = filtered_speaker[0]
            duration_full = float(seg["end"]) - float(seg["start"])
            duration_full = max(1.0, min(duration_full, 18.0))

            name_new_wav = str(seg["start"])
            create_wav_file_vc(
                sample_name=name_new_wav,
                audio_wav="audio.wav",
                start=(float(seg["start"])),
                end=(float(seg["start"]) + duration_full),
                output_final_path=dir_path_speaker,
                get_vocals_dereverb=get_vocals_dereverb,
            )

            file_name_tts = f"audio2/audio/{str(seg['start'])}.ogg"
            convert_to_xtts_good_sample(file_name_tts, dir_path_speaker_tts)

    logger.debug(f"Base(source_ref_tts): {str(path_source_segments)}")
    logger.debug(f"Target(orig_voice): {str(path_target_segments)}")

    return path_source_segments, path_target_segments


def toneconverter_openvoice(
    result_diarize,
    preprocessor_max_segments,
    remove_previous_process=True,
    get_vocals_dereverb=False,
    model="openvoice",
):
    audio_path = "audio.wav"
    target_dir = "processed"
    create_directories(target_dir)

    from openvoice import se_extractor
    from openvoice.api import ToneColorConverter

    audio_name = (
        f"{os.path.basename(audio_path).rsplit('.', 1)[0]}_"
        f"{se_extractor.hash_numpy_array(audio_path)}"
    )

    valid_speakers = list(
        {item["speaker"] for item in result_diarize["segments"]}
    )

    logger.info("Openvoice preprocessor...")

    if remove_previous_process:
        remove_directory_contents(target_dir)

    path_source_segments, path_target_segments = create_wav_vc(
        valid_speakers,
        result_diarize["segments"],
        audio_name,
        max_segments=preprocessor_max_segments,
        get_vocals_dereverb=get_vocals_dereverb,
    )

    logger.info("Openvoice loading model...")
    model_path_openvoice = "./OPENVOICE_MODELS"
    url_model_openvoice = (
        "https://huggingface.co/myshell-ai/OpenVoice/resolve/main/checkpoints/converter"
    )

    if "v2" in model:
        model_path = os.path.join(model_path_openvoice, "v2")
        url_model_openvoice = url_model_openvoice.replace(
            "OpenVoice", "OpenVoiceV2"
        ).replace("checkpoints/", "")
    else:
        model_path = os.path.join(model_path_openvoice, "v1")
    create_directories(model_path)

    config_url = f"{url_model_openvoice}/config.json"
    checkpoint_url = f"{url_model_openvoice}/checkpoint.pth"

    config_path = download_manager(url=config_url, path=model_path)
    checkpoint_path = download_manager(
        url=checkpoint_url, path=model_path
    )

    device = os.environ.get("SONITR_DEVICE")
    tone_color_converter = ToneColorConverter(config_path, device=device)
    tone_color_converter.load_ckpt(checkpoint_path)

    logger.info("Openvoice tone color converter:")
    global_progress_bar = tqdm(
        total=len(result_diarize["segments"]), desc="Progress"
    )

    for source_seg, target_seg, speaker in zip(
        path_source_segments, path_target_segments, valid_speakers
    ):
        source_se = se_process_audio_segments(
            source_seg, tone_color_converter, device
        )
        target_se = se_process_audio_segments(
            target_seg, tone_color_converter, device
        )

        encode_message = "@MyShell"

        filtered_speaker = [
            segment
            for segment in result_diarize["segments"]
            if segment["speaker"] == speaker
        ]
        for seg in filtered_speaker:
            src_path = save_path = f"audio2/audio/{str(seg['start'])}.ogg"

            tone_color_converter.convert(
                audio_src_path=src_path,
                src_se=source_se,
                tgt_se=target_se,
                output_path=save_path,
                message=encode_message,
            )

            global_progress_bar.update(1)

    global_progress_bar.close()

    try:
        del tone_color_converter
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as error:
        logger.error(str(error))
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def toneconverter_freevc(
    result_diarize,
    remove_previous_process=True,
    get_vocals_dereverb=False,
):
    audio_path = "audio.wav"
    target_dir = "processed"
    create_directories(target_dir)

    from openvoice import se_extractor

    audio_name = (
        f"{os.path.basename(audio_path).rsplit('.', 1)[0]}_"
        f"{se_extractor.hash_numpy_array(audio_path)}"
    )

    valid_speakers = list(
        {item["speaker"] for item in result_diarize["segments"]}
    )

    logger.info("FreeVC preprocessor...")

    if remove_previous_process:
        remove_directory_contents(target_dir)

    path_source_segments, path_target_segments = create_wav_vc(
        valid_speakers,
        result_diarize["segments"],
        audio_name,
        max_segments=1,
        get_vocals_dereverb=get_vocals_dereverb,
    )

    logger.info("FreeVC loading model...")
    device_id = os.environ.get("SONITR_DEVICE")
    device = None if device_id == "cpu" else device_id
    try:
        # тут мы выкарыстоўваем іншы TTS API для голасавай канверсіі
        from TTS.api import TTS  # гэта інш. мадэль freevc24

        tts = TTS(
            model_name="voice_conversion_models/multilingual/vctk/freevc24",
            progress_bar=False,
        ).to(device)
    except Exception as error:
        logger.error(str(error))
        logger.error("Error loading the FreeVC model.")
        return

    logger.info("FreeVC process:")
    global_progress_bar = tqdm(
        total=len(result_diarize["segments"]), desc="Progress"
    )

    for source_seg, target_seg, speaker in zip(
        path_source_segments, path_target_segments, valid_speakers
    ):

        filtered_speaker = [
            segment
            for segment in result_diarize["segments"]
            if segment["speaker"] == speaker
        ]

        files_and_directories = os.listdir(target_seg)
        wav_files = [
            file for file in files_and_directories if file.endswith(".wav")
        ]
        original_wav_audio_segment = os.path.join(target_seg, wav_files[0])

        for seg in filtered_speaker:

            src_path = save_path = f"audio2/audio/{str(seg['start'])}.ogg"
            logger.debug(f"{src_path} - {original_wav_audio_segment}")

            wav = tts.voice_conversion(
                source_wav=src_path,
                target_wav=original_wav_audio_segment,
            )

            write_chunked(
                file=save_path,
                samplerate=tts.voice_converter.vc_config.audio.output_sample_rate,
                data=wav,
                format="ogg",
                subtype="vorbis",
            )

            global_progress_bar.update(1)

    global_progress_bar.close()

    try:
        del tts
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as error:
        logger.error(str(error))
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def toneconverter(
    result_diarize,
    preprocessor_max_segments,
    remove_previous_process=True,
    get_vocals_dereverb=False,
    method_vc="freevc"
):
    if method_vc == "freevc":
        if preprocessor_max_segments > 1:
            logger.info("FreeVC only uses one segment.")
        return toneconverter_freevc(
            result_diarize,
            remove_previous_process=remove_previous_process,
            get_vocals_dereverb=get_vocals_dereverb,
        )
    elif "openvoice" in method_vc:
        return toneconverter_openvoice(
            result_diarize,
            preprocessor_max_segments,
            remove_previous_process=remove_previous_process,
            get_vocals_dereverb=get_vocals_dereverb,
            model=method_vc,
        )


if __name__ == "__main__":
    # лакальны хуткі тэст (калі існуе segments.py з result_diarize)
    from segments import result_diarize

    audio_segmentation_to_voice(
        result_diarize,
        TRANSLATE_AUDIO_TO="be",
        is_gui=True,
        tts_voice00="_XTTS_/AUTOMATIC.wav",
    )
