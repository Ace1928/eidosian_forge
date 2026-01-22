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
def get_dc_offset(self, channel=1):
    """
        Returns a value between -1.0 and 1.0 representing the DC offset of a
        channel (1 for left, 2 for right).
        """
    if not 1 <= channel <= 2:
        raise ValueError('channel value must be 1 (left) or 2 (right)')
    if self.channels == 1:
        data = self._data
    elif channel == 1:
        data = audioop.tomono(self._data, self.sample_width, 1, 0)
    else:
        data = audioop.tomono(self._data, self.sample_width, 0, 1)
    return float(audioop.avg(data, self.sample_width)) / self.max_possible_amplitude