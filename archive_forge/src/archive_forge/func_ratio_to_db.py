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
def ratio_to_db(ratio, val2=None, using_amplitude=True):
    """
    Converts the input float to db, which represents the equivalent
    to the ratio in power represented by the multiplier passed in.
    """
    ratio = float(ratio)
    if val2 is not None:
        ratio = ratio / val2
    if ratio == 0:
        return -float('inf')
    if using_amplitude:
        return 20 * log(ratio, 10)
    else:
        return 10 * log(ratio, 10)