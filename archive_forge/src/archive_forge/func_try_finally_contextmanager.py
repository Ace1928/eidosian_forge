from __future__ import absolute_import
import cython
import os
import sys
import re
import io
import codecs
import glob
import shutil
import tempfile
from functools import wraps
from . import __version__ as cython_version
def try_finally_contextmanager(gen_func):

    @wraps(gen_func)
    def make_gen(*args, **kwargs):
        return _TryFinallyGeneratorContextManager(gen_func(*args, **kwargs))
    return make_gen