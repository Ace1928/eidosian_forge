from __future__ import print_function
import gdb
import os
import locale
import sys
import sys
import libpython
import re
import warnings
import tempfile
import functools
import textwrap
import itertools
import traceback
@classmethod
def get_selected_bytecode_frame(cls):
    """Try to obtain the Frame for the python bytecode interpreter in the
        selected GDB frame, or None"""
    frame = cls.get_selected_frame()
    while frame:
        if frame.is_evalframe():
            return frame
        frame = frame.older()
    return None