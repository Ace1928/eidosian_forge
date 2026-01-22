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
def fix_wav_headers(data):
    headers = extract_wav_headers(data)
    if not headers or headers[-1].id != b'data':
        return
    if len(data) > 2 ** 32:
        raise CouldntDecodeError('Unable to process >4GB files')
    data[4:8] = struct.pack('<I', len(data) - 8)
    pos = headers[-1].position
    data[pos + 4:pos + 8] = struct.pack('<I', len(data) - pos - 8)