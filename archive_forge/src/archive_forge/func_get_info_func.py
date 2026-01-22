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
def get_info_func():
    backends = get_available_backends()

    def dispatcher(uri: Union[BinaryIO, str, os.PathLike], format: Optional[str], backend_name: Optional[str]) -> Backend:
        if backend_name is not None:
            return get_backend(backend_name, backends)
        for backend in backends.values():
            if backend.can_decode(uri, format):
                return backend
        raise RuntimeError(f"Couldn't find appropriate backend to handle uri {uri} and format {format}.")

    def info(uri: Union[BinaryIO, str, os.PathLike], format: Optional[str]=None, buffer_size: int=4096, backend: Optional[str]=None) -> AudioMetaData:
        """Get signal information of an audio file.

        Note:
            When the input type is file-like object, this function cannot
            get the correct length (``num_samples``) for certain formats,
            such as ``vorbis``.
            In this case, the value of ``num_samples`` is ``0``.

        Args:
            uri (path-like object or file-like object):
                Source of audio data. The following types are accepted:

                * ``path-like``: File path or URL.
                * ``file-like``: Object with ``read(size: int) -> bytes`` method,
                  which returns byte string of at most ``size`` length.

            format (str or None, optional):
                If not ``None``, interpreted as hint that may allow backend to override the detected format.
                (Default: ``None``)

            buffer_size (int, optional):
                Size of buffer to use when processing file-like objects, in bytes. (Default: ``4096``)

            backend (str or None, optional):
                I/O backend to use.
                If ``None``, function selects backend given input and available backends.
                Otherwise, must be one of [``"ffmpeg"``, ``"sox"``, ``"soundfile"``],
                with the corresponding backend available.
                (Default: ``None``)

                .. seealso::
                   :ref:`backend`

        Returns:
            AudioMetaData
        """
        backend = dispatcher(uri, format, backend)
        return backend.info(uri, format, buffer_size)
    return info