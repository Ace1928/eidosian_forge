import os
import re
import sys
from typing import BinaryIO, Optional, Tuple, Union
import torch
import torchaudio
from .backend import Backend
from .common import AudioMetaData
def load_audio(src: InputType, frame_offset: int=0, num_frames: int=-1, convert: bool=True, channels_first: bool=True, format: Optional[str]=None, buffer_size: int=4096) -> Tuple[torch.Tensor, int]:
    if hasattr(src, 'read') and format == 'vorbis':
        format = 'ogg'
    s = torchaudio.io.StreamReader(src, format, None, buffer_size)
    sample_rate = int(s.get_src_stream_info(s.default_audio_stream).sample_rate)
    filter = _get_load_filter(frame_offset, num_frames, convert)
    waveform = _load_audio(s, filter, channels_first)
    return (waveform, sample_rate)