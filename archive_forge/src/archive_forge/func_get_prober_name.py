from __future__ import division
import json
import os
import re
import sys
from subprocess import Popen, PIPE
from math import log, ceil
from tempfile import TemporaryFile
from warnings import warn
from functools import wraps
def get_prober_name():
    """
    Return probe application, either avconv or ffmpeg
    """
    if which('avprobe'):
        return 'avprobe'
    elif which('ffprobe'):
        return 'ffprobe'
    else:
        warn("Couldn't find ffprobe or avprobe - defaulting to ffprobe, but may not work", RuntimeWarning)
        return 'ffprobe'