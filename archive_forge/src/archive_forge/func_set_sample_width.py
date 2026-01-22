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
def set_sample_width(self, sample_width):
    if sample_width == self.sample_width:
        return self
    frame_width = self.channels * sample_width
    return self._spawn(audioop.lin2lin(self._data, self.sample_width, sample_width), overrides={'sample_width': sample_width, 'frame_width': frame_width})