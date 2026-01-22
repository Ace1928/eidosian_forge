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
def remove_dc_offset(self, channel=None, offset=None):
    """
        Removes DC offset of given channel. Calculates offset if it's not given.
        Offset values must be in range -1.0 to 1.0. If channel is None, removes
        DC offset from all available channels.
        """
    if channel and (not 1 <= channel <= 2):
        raise ValueError('channel value must be None, 1 (left) or 2 (right)')
    if offset and (not -1.0 <= offset <= 1.0):
        raise ValueError('offset value must be in range -1.0 to 1.0')
    if offset:
        offset = int(round(offset * self.max_possible_amplitude))

    def remove_data_dc(data, off):
        if not off:
            off = audioop.avg(data, self.sample_width)
        return audioop.bias(data, self.sample_width, -off)
    if self.channels == 1:
        return self._spawn(data=remove_data_dc(self._data, offset))
    left_channel = audioop.tomono(self._data, self.sample_width, 1, 0)
    right_channel = audioop.tomono(self._data, self.sample_width, 0, 1)
    if not channel or channel == 1:
        left_channel = remove_data_dc(left_channel, offset)
    if not channel or channel == 2:
        right_channel = remove_data_dc(right_channel, offset)
    left_channel = audioop.tostereo(left_channel, self.sample_width, 1, 0)
    right_channel = audioop.tostereo(right_channel, self.sample_width, 0, 1)
    return self._spawn(data=audioop.add(left_channel, right_channel, self.sample_width))