import ast
import os
import re
from hacking import core
from os_win.utils.winapi import libs as w_lib
import_translation_for_log_or_exception = re.compile(
@core.flake8ext
def no_translate_logs(logical_line):
    """Check for 'LOG.*(_('

    Starting with the Pike series, OpenStack no longer supports log
    translation. We shouldn't translate logs.

    - This check assumes that 'LOG' is a logger.
    - Use filename so we can start enforcing this in specific folders
      instead of needing to do so all at once.

    C312
    """
    if _log_translation_hint.match(logical_line):
        yield (0, 'C312: Log messages should not be translated!')