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
def get_encoder_name():
    """
    Return enconder default application for system, either avconv or ffmpeg
    """
    if which('avconv'):
        return 'avconv'
    elif which('ffmpeg'):
        return 'ffmpeg'
    else:
        warn("Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work", RuntimeWarning)
        return 'ffmpeg'