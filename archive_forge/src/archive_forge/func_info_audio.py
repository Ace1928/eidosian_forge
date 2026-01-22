import os
import re
import sys
from typing import BinaryIO, Optional, Tuple, Union
import torch
import torchaudio
from .backend import Backend
from .common import AudioMetaData
def info_audio(src: InputType, format: Optional[str], buffer_size: int=4096) -> AudioMetaData:
    s = torchaudio.io.StreamReader(src, format, None, buffer_size)
    sinfo = s.get_src_stream_info(s.default_audio_stream)
    if sinfo.num_frames == 0:
        waveform = _load_audio(s)
        num_frames = waveform.size(1)
    else:
        num_frames = sinfo.num_frames
    return AudioMetaData(int(sinfo.sample_rate), num_frames, sinfo.num_channels, sinfo.bits_per_sample, sinfo.codec.upper())