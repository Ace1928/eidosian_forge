import os
from functools import lru_cache
from typing import BinaryIO, Dict, Optional, Tuple, Type, Union
import torch
from torchaudio._extension import lazy_import_sox_ext
from torchaudio.io import CodecConfig
from torio._extension import lazy_import_ffmpeg_ext
from . import soundfile_backend
from .backend import Backend
from .common import AudioMetaData
from .ffmpeg import FFmpegBackend
from .soundfile import SoundfileBackend
from .sox import SoXBackend
@lru_cache(None)
def get_available_backends() -> Dict[str, Type[Backend]]:
    backend_specs: Dict[str, Type[Backend]] = {}
    if lazy_import_ffmpeg_ext().is_available():
        backend_specs['ffmpeg'] = FFmpegBackend
    if lazy_import_sox_ext().is_available():
        backend_specs['sox'] = SoXBackend
    if soundfile_backend._IS_SOUNDFILE_AVAILABLE:
        backend_specs['soundfile'] = SoundfileBackend
    return backend_specs