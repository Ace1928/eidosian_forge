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
def ffmpeg_get_packet_pts(file, packet):
    if packet.contents.dts != AV_NOPTS_VALUE:
        pts = packet.contents.dts
    else:
        pts = 0
    timestamp = avutil.av_rescale_q(pts, file.context.contents.streams[packet.contents.stream_index].contents.time_base, AV_TIME_BASE_Q)
    return timestamp