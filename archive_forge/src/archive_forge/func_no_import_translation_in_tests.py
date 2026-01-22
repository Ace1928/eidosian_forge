import ast
import os
import re
from hacking import core
from os_win.utils.winapi import libs as w_lib
import_translation_for_log_or_exception = re.compile(
@core.flake8ext
def no_import_translation_in_tests(logical_line, filename):
    """Check for 'from os_win._i18n import _'

    N337
    """
    if 'os_win/tests/' in filename:
        res = import_translation_for_log_or_exception.match(logical_line)
        if res:
            yield (0, "N337 Don't import translation in tests")