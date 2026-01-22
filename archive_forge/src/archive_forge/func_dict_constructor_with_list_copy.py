import ast
import os
import re
from hacking import core
from os_win.utils.winapi import libs as w_lib
import_translation_for_log_or_exception = re.compile(
@core.flake8ext
def dict_constructor_with_list_copy(logical_line):
    msg = 'N336: Must use a dict comprehension instead of a dict constructor with a sequence of key-value pairs.'
    if dict_constructor_with_list_copy_re.match(logical_line):
        yield (0, msg)