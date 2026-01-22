import sys
from collections import deque
from ctypes import (c_int, c_int32, c_uint8, c_char_p,
import pyglet
import pyglet.lib
from pyglet import image
from pyglet.util import asbytes, asstr
from . import MediaDecoder
from .base import AudioData, SourceInfo, StaticSource
from .base import StreamingSource, VideoFormat, AudioFormat
from .ffmpeg_lib import *
from ..exceptions import MediaFormatException
def start_times(streams):
    yield 0
    for stream in streams:
        start = stream.start_time
        if start == AV_NOPTS_VALUE:
            yield 0
        start_time = avutil.av_rescale_q(start, stream.time_base, AV_TIME_BASE_Q)
        start_time = timestamp_from_ffmpeg(start_time)
        yield start_time