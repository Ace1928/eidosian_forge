import warnings
from typing import Optional, Tuple
import torch
from torchaudio._internal import module_utils as _mod_utils
from .common import AudioMetaData
Save audio data to file.

    Note:
        The formats this function can handle depend on the soundfile installation.
        This function is tested on the following formats;

        * WAV

            * 32-bit floating-point
            * 32-bit signed integer
            * 16-bit signed integer
            * 8-bit unsigned integer

        * FLAC
        * OGG/VORBIS
        * SPHERE

    Note:
        ``filepath`` argument is intentionally annotated as ``str`` only, even though it accepts
        ``pathlib.Path`` object as well. This is for the consistency with ``"sox_io"`` backend,
        which has a restriction on type annotation due to TorchScript compiler compatiblity.

    Args:
        filepath (str or pathlib.Path): Path to audio file.
        src (torch.Tensor): Audio data to save. must be 2D tensor.
        sample_rate (int): sampling rate
        channels_first (bool, optional): If ``True``, the given tensor is interpreted as `[channel, time]`,
            otherwise `[time, channel]`.
        compression (float of None, optional): Not used.
            It is here only for interface compatibility reson with "sox_io" backend.
        format (str or None, optional): Override the audio format.
            When ``filepath`` argument is path-like object, audio format is
            inferred from file extension. If the file extension is missing or
            different, you can specify the correct format with this argument.

            When ``filepath`` argument is file-like object,
            this argument is required.

            Valid values are ``"wav"``, ``"ogg"``, ``"vorbis"``,
            ``"flac"`` and ``"sph"``.
        encoding (str or None, optional): Changes the encoding for supported formats.
            This argument is effective only for supported formats, sush as
            ``"wav"``, ``""flac"`` and ``"sph"``. Valid values are;

                - ``"PCM_S"`` (signed integer Linear PCM)
                - ``"PCM_U"`` (unsigned integer Linear PCM)
                - ``"PCM_F"`` (floating point PCM)
                - ``"ULAW"`` (mu-law)
                - ``"ALAW"`` (a-law)

        bits_per_sample (int or None, optional): Changes the bit depth for the
            supported formats.
            When ``format`` is one of ``"wav"``, ``"flac"`` or ``"sph"``,
            you can change the bit depth.
            Valid values are ``8``, ``16``, ``24``, ``32`` and ``64``.

    Supported formats/encodings/bit depth/compression are:

    ``"wav"``
        - 32-bit floating-point PCM
        - 32-bit signed integer PCM
        - 24-bit signed integer PCM
        - 16-bit signed integer PCM
        - 8-bit unsigned integer PCM
        - 8-bit mu-law
        - 8-bit a-law

        Note:
            Default encoding/bit depth is determined by the dtype of
            the input Tensor.

    ``"flac"``
        - 8-bit
        - 16-bit (default)
        - 24-bit

    ``"ogg"``, ``"vorbis"``
        - Doesn't accept changing configuration.

    ``"sph"``
        - 8-bit signed integer PCM
        - 16-bit signed integer PCM
        - 24-bit signed integer PCM
        - 32-bit signed integer PCM (default)
        - 8-bit mu-law
        - 8-bit a-law
        - 16-bit a-law
        - 24-bit a-law
        - 32-bit a-law

    