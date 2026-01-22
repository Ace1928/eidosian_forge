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
def get_player_name():
    """
    Return enconder default application for system, either avconv or ffmpeg
    """
    if which('avplay'):
        return 'avplay'
    elif which('ffplay'):
        return 'ffplay'
    else:
        warn("Couldn't find ffplay or avplay - defaulting to ffplay, but may not work", RuntimeWarning)
        return 'ffplay'