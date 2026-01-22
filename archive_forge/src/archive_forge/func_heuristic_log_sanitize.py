from __future__ import absolute_import, division, print_function
import sys
import __main__
import atexit
import errno
import datetime
import grp
import fcntl
import locale
import os
import pwd
import platform
import re
import select
import shlex
import shutil
import signal
import stat
import subprocess
import tempfile
import time
import traceback
import types
from itertools import chain, repeat
from ansible.module_utils.compat import selectors
from ._text import to_native, to_bytes, to_text
from ansible.module_utils.common.text.converters import (
from ansible.module_utils.common.arg_spec import ModuleArgumentSpecValidator
from ansible.module_utils.common.text.formatters import (
import hashlib
from ansible.module_utils.six.moves.collections_abc import (
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.file import (
from ansible.module_utils.common.sys_info import (
from ansible.module_utils.pycompat24 import get_exception, literal_eval
from ansible.module_utils.common.parameters import (
from ansible.module_utils.errors import AnsibleFallbackNotFound, AnsibleValidationErrorMultiple, UnsupportedError
from ansible.module_utils.six import (
from ansible.module_utils.six.moves import map, reduce, shlex_quote
from ansible.module_utils.common.validation import (
from ansible.module_utils.common._utils import get_all_subclasses as _get_all_subclasses
from ansible.module_utils.parsing.convert_bool import BOOLEANS, BOOLEANS_FALSE, BOOLEANS_TRUE, boolean
from ansible.module_utils.common.warnings import (
def heuristic_log_sanitize(data, no_log_values=None):
    """ Remove strings that look like passwords from log messages """
    data = to_native(data)
    output = []
    begin = len(data)
    prev_begin = begin
    sep = 1
    while sep:
        try:
            end = data.rindex('@', 0, begin)
        except ValueError:
            output.insert(0, data[0:begin])
            break
        sep = None
        sep_search_end = end
        while not sep:
            try:
                begin = data.rindex('://', 0, sep_search_end)
            except ValueError:
                begin = 0
            try:
                sep = data.index(':', begin + 3, end)
            except ValueError:
                if begin == 0:
                    output.insert(0, data[0:prev_begin])
                    break
                sep_search_end = begin
                continue
        if sep:
            output.insert(0, data[end:prev_begin])
            output.insert(0, '********')
            output.insert(0, data[begin:sep + 1])
            prev_begin = begin
    output = ''.join(output)
    if no_log_values:
        output = remove_values(output, no_log_values)
    return output