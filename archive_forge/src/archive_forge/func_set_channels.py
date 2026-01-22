from __future__ import division
import array
import os
import subprocess
from tempfile import TemporaryFile, NamedTemporaryFile
import wave
import sys
import struct
from .logging_utils import log_conversion, log_subprocess_output
from .utils import mediainfo_json, fsdecode
import base64
from collections import namedtuple
from io import BytesIO
from .utils import (
from .exceptions import (
from . import effects
def set_channels(self, channels):
    if channels == self.channels:
        return self
    if channels == 2 and self.channels == 1:
        fn = audioop.tostereo
        frame_width = self.frame_width * 2
        fac = 1
        converted = fn(self._data, self.sample_width, fac, fac)
    elif channels == 1 and self.channels == 2:
        fn = audioop.tomono
        frame_width = self.frame_width // 2
        fac = 0.5
        converted = fn(self._data, self.sample_width, fac, fac)
    elif channels == 1:
        channels_data = [seg.get_array_of_samples() for seg in self.split_to_mono()]
        frame_count = int(self.frame_count())
        converted = array.array(channels_data[0].typecode, b'\x00' * (frame_count * self.sample_width))
        for raw_channel_data in channels_data:
            for i in range(frame_count):
                converted[i] += raw_channel_data[i] // self.channels
        frame_width = self.frame_width // self.channels
    elif self.channels == 1:
        dup_channels = [self for iChannel in range(channels)]
        return AudioSegment.from_mono_audiosegments(*dup_channels)
    else:
        raise ValueError('AudioSegment.set_channels only supports mono-to-multi channel and multi-to-mono channel conversion')
    return self._spawn(data=converted, overrides={'channels': channels, 'frame_width': frame_width})