import os
import sys
import traceback
from contextlib import contextmanager
from functools import partial
from pprint import pprint
from celery.platforms import signals
from celery.utils.text import WhateverIO
def humanbytes(s):
    """Convert bytes to human-readable form (e.g., KB, MB)."""
    return next((f'{hfloat(s / div if div else s)}{unit}' for div, unit in UNITS if s >= div))