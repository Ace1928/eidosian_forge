from __future__ import annotations
import aifc
import audioop
import base64
import collections
import hashlib
import hmac
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from .audio import AudioData, get_flac_converter
from .exceptions import (
def threaded_listen():
    with source as s:
        while running[0]:
            try:
                audio = self.listen(s, 1, phrase_time_limit)
            except WaitTimeoutError:
                pass
            else:
                if running[0]:
                    callback(self, audio)