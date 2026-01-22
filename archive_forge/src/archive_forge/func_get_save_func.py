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
def get_save_func():
    backends = get_available_backends()

    def dispatcher(uri: Union[BinaryIO, str, os.PathLike], format: Optional[str], backend_name: Optional[str]) -> Backend:
        if backend_name is not None:
            return get_backend(backend_name, backends)
        for backend in backends.values():
            if backend.can_encode(uri, format):
                return backend
        raise RuntimeError(f"Couldn't find appropriate backend to handle uri {uri} and format {format}.")

    def save(uri: Union[BinaryIO, str, os.PathLike], src: torch.Tensor, sample_rate: int, channels_first: bool=True, format: Optional[str]=None, encoding: Optional[str]=None, bits_per_sample: Optional[int]=None, buffer_size: int=4096, backend: Optional[str]=None, compression: Optional[Union[CodecConfig, float, int]]=None):
        """Save audio data to file.

        Note:
            The formats this function can handle depend on the availability of backends.
            Please use the following functions to fetch the supported formats.

            - FFmpeg: :py:func:`torchaudio.utils.ffmpeg_utils.get_audio_encoders`
            - Sox: :py:func:`torchaudio.utils.sox_utils.list_write_formats`
            - SoundFile: Refer to `the official document <https://pysoundfile.readthedocs.io/>`__.

        Args:
            uri (str or pathlib.Path): Path to audio file.
            src (torch.Tensor): Audio data to save. must be 2D tensor.
            sample_rate (int): sampling rate
            channels_first (bool, optional): If ``True``, the given tensor is interpreted as `[channel, time]`,
                otherwise `[time, channel]`.
            format (str or None, optional): Override the audio format.
                When ``uri`` argument is path-like object, audio format is
                inferred from file extension. If the file extension is missing or
                different, you can specify the correct format with this argument.

                When ``uri`` argument is file-like object,
                this argument is required.

                Valid values are ``"wav"``, ``"ogg"``, and ``"flac"``.
            encoding (str or None, optional): Changes the encoding for supported formats.
                This argument is effective only for supported formats, i.e.
                ``"wav"`` and ``""flac"```. Valid values are

                - ``"PCM_S"`` (signed integer Linear PCM)
                - ``"PCM_U"`` (unsigned integer Linear PCM)
                - ``"PCM_F"`` (floating point PCM)
                - ``"ULAW"`` (mu-law)
                - ``"ALAW"`` (a-law)

            bits_per_sample (int or None, optional): Changes the bit depth for the
                supported formats.
                When ``format`` is one of ``"wav"`` and ``"flac"``,
                you can change the bit depth.
                Valid values are ``8``, ``16``, ``24``, ``32`` and ``64``.

            buffer_size (int, optional):
                Size of buffer to use when processing file-like objects, in bytes. (Default: ``4096``)

            backend (str or None, optional):
                I/O backend to use.
                If ``None``, function selects backend given input and available backends.
                Otherwise, must be one of [``"ffmpeg"``, ``"sox"``, ``"soundfile"``],
                with the corresponding backend being available.
                (Default: ``None``)

                .. seealso::
                   :ref:`backend`

            compression (CodecConfig, float, int, or None, optional):
                Compression configuration to apply.

                If the selected backend is FFmpeg, an instance of :py:class:`CodecConfig` must be provided.

                Otherwise, if the selected backend is SoX, a float or int value corresponding to option ``-C`` of the
                ``sox`` command line interface must be provided. For instance:

                ``"mp3"``
                    Either bitrate (in ``kbps``) with quality factor, such as ``128.2``, or
                    VBR encoding with quality factor such as ``-4.2``. Default: ``-4.5``.

                ``"flac"``
                    Whole number from ``0`` to ``8``. ``8`` is default and highest compression.

                ``"ogg"``, ``"vorbis"``
                    Number from ``-1`` to ``10``; ``-1`` is the highest compression
                    and lowest quality. Default: ``3``.

                Refer to http://sox.sourceforge.net/soxformat.html for more details.

        """
        backend = dispatcher(uri, format, backend)
        return backend.save(uri, src, sample_rate, channels_first, format, encoding, bits_per_sample, buffer_size, compression)
    return save