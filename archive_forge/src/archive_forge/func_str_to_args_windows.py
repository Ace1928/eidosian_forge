import os
import re
import sys
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_constants import get_global_debugger, IS_WINDOWS, IS_JYTHON, get_current_thread_id, \
from _pydev_bundle import pydev_log
from contextlib import contextmanager
from _pydevd_bundle import pydevd_constants, pydevd_defaults
from _pydevd_bundle.pydevd_defaults import PydevdCustomization
import ast
def str_to_args_windows(args):
    result = []
    DEFAULT = 0
    ARG = 1
    IN_DOUBLE_QUOTE = 2
    state = DEFAULT
    backslashes = 0
    buf = ''
    args_len = len(args)
    for i in range(args_len):
        ch = args[i]
        if ch == '\\':
            backslashes += 1
            continue
        elif backslashes != 0:
            if ch == '"':
                while backslashes >= 2:
                    backslashes -= 2
                    buf += '\\'
                if backslashes == 1:
                    if state == DEFAULT:
                        state = ARG
                    buf += '"'
                    backslashes = 0
                    continue
            else:
                if state == DEFAULT:
                    state = ARG
                while backslashes > 0:
                    backslashes -= 1
                    buf += '\\'
        if ch in (' ', '\t'):
            if state == DEFAULT:
                continue
            elif state == ARG:
                state = DEFAULT
                result.append(buf)
                buf = ''
                continue
        if state in (DEFAULT, ARG):
            if ch == '"':
                state = IN_DOUBLE_QUOTE
            else:
                state = ARG
                buf += ch
        elif state == IN_DOUBLE_QUOTE:
            if ch == '"':
                if i + 1 < args_len and args[i + 1] == '"':
                    buf += '"'
                    i += 1
                else:
                    state = ARG
            else:
                buf += ch
        else:
            raise RuntimeError('Illegal condition')
    if len(buf) > 0 or state != DEFAULT:
        result.append(buf)
    return result