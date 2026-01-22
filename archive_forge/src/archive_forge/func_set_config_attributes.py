import os
import platform
import warnings
from pyglet import image
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32 import com
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.media import Source
from pyglet.media.codecs import AudioFormat, AudioData, VideoFormat, MediaDecoder, StaticSource
from pyglet.util import debug_print, DecodeException
def set_config_attributes(self):
    """ Here we set user specified attributes, by default we try to set low latency mode. (Win7+)"""
    if self.low_latency or self.decode_video:
        self._attributes = IMFAttributes()
        MFCreateAttributes(ctypes.byref(self._attributes), 3)
    if self.low_latency and WINDOWS_7_OR_GREATER:
        self._attributes.SetUINT32(ctypes.byref(MF_LOW_LATENCY), 1)
        assert _debug('WMFAudioDecoder: Setting configuration attributes.')
    if self.decode_video:
        self._attributes.SetUINT32(ctypes.byref(MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS), 1)
        self._attributes.SetUINT32(ctypes.byref(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING), 1)
        assert _debug('WMFVideoDecoder: Setting configuration attributes.')