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
@classmethod
def silent(cls, duration=1000, frame_rate=11025):
    """
        Generate a silent audio segment.
        duration specified in milliseconds (default duration: 1000ms, default frame_rate: 11025).
        """
    frames = int(frame_rate * (duration / 1000.0))
    data = b'\x00\x00' * frames
    return cls(data, metadata={'channels': 1, 'sample_width': 2, 'frame_rate': frame_rate, 'frame_width': 2})