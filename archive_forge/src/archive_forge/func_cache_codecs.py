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
def cache_codecs(function):
    cache = {}

    @wraps(function)
    def wrapper():
        try:
            return cache[0]
        except:
            cache[0] = function()
            return cache[0]
    return wrapper