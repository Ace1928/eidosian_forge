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
def get_array_of_samples(self, array_type_override=None):
    """
        returns the raw_data as an array of samples
        """
    if array_type_override is None:
        array_type_override = self.array_type
    return array.array(array_type_override, self._data)