import cffi  # type: ignore
import os
import re
import sys
import warnings
import situation  # preloaded in setup.py
import importlib
def define_rlen_kind(ffibuilder, definitions):
    if ffibuilder.sizeof('size_t') > 4:
        definitions['RPY2_RLEN_LONG'] = True