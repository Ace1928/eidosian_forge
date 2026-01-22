import ast
import os
import re
from hacking import core
from os_win.utils.winapi import libs as w_lib
import_translation_for_log_or_exception = re.compile(
@core.flake8ext
def check_explicit_underscore_import(logical_line, filename):
    """Check for explicit import of the _ function

    We need to ensure that any files that are using the _() function
    to translate logs are explicitly importing the _ function.  We
    can't trust unit test to catch whether the import has been
    added so we need to check for it here.
    """
    if filename in UNDERSCORE_IMPORT_FILES:
        pass
    elif underscore_import_check.match(logical_line) or custom_underscore_check.match(logical_line):
        UNDERSCORE_IMPORT_FILES.append(filename)
    elif string_translation.match(logical_line):
        yield (0, 'N323: Found use of _() without explicit import of _ !')