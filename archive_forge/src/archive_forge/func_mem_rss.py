import os
import sys
import traceback
from contextlib import contextmanager
from functools import partial
from pprint import pprint
from celery.platforms import signals
from celery.utils.text import WhateverIO
def mem_rss():
    """Return RSS memory usage as a humanized string."""
    p = ps()
    if p is not None:
        return humanbytes(_process_memory_info(p).rss)