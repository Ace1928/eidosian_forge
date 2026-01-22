import ast
import os
import re
from hacking import core
from os_win.utils.winapi import libs as w_lib
import_translation_for_log_or_exception = re.compile(
@core.flake8ext
def use_timeutils_utcnow(logical_line, filename):
    if '/tools/' in filename:
        return
    msg = 'N310: timeutils.utcnow() must be used instead of datetime.%s()'
    datetime_funcs = ['now', 'utcnow']
    for f in datetime_funcs:
        pos = logical_line.find('datetime.%s' % f)
        if pos != -1:
            yield (pos, msg % f)