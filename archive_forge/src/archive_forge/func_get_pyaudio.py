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
@staticmethod
def get_pyaudio():
    """
        Imports the pyaudio module and checks its version. Throws exceptions if pyaudio can't be found or a wrong version is installed
        """
    try:
        import pyaudio
    except ImportError:
        raise AttributeError('Could not find PyAudio; check installation')
    from distutils.version import LooseVersion
    if LooseVersion(pyaudio.__version__) < LooseVersion('0.2.11'):
        raise AttributeError('PyAudio 0.2.11 or later is required (found version {})'.format(pyaudio.__version__))
    return pyaudio