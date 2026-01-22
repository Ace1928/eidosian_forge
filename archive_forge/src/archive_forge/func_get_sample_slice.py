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
def get_sample_slice(self, start_sample=None, end_sample=None):
    """
        Get a section of the audio segment by sample index.

        NOTE: Negative indices do *not* address samples backword
        from the end of the audio segment like a python list.
        This is intentional.
        """
    max_val = int(self.frame_count())

    def bounded(val, default):
        if val is None:
            return default
        if val < 0:
            return 0
        if val > max_val:
            return max_val
        return val
    start_i = bounded(start_sample, 0) * self.frame_width
    end_i = bounded(end_sample, max_val) * self.frame_width
    data = self._data[start_i:end_i]
    return self._spawn(data)