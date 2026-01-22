import ast
import os
import re
from hacking import core
from os_win.utils.winapi import libs as w_lib
import_translation_for_log_or_exception = re.compile(
@core.flake8ext
def assert_raises_regexp(logical_line):
    """Check for usage of deprecated assertRaisesRegexp

    N335
    """
    res = asse_raises_regexp.search(logical_line)
    if res:
        yield (0, 'N335: assertRaisesRegex must be used instead of assertRaisesRegexp')