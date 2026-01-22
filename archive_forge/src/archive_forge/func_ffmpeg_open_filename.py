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
def ffmpeg_open_filename(filename):
    """Open the media file.

    :rtype: FFmpegFile
    :return: The structure containing all the information for the media.
    """
    file = FFmpegFile()
    result = avformat.avformat_open_input(byref(file.context), filename, None, None)
    if result != 0:
        raise FFmpegException('avformat_open_input in ffmpeg_open_filename returned an error opening file ' + filename.decode('utf8') + ' Error code: ' + str(result))
    result = avformat.avformat_find_stream_info(file.context, None)
    if result < 0:
        raise FFmpegException('Could not find stream info')
    return file