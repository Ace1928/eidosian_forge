import re
from hacking import core
@core.flake8ext
def check_no_contextlib_nested(logical_line):
    msg = 'G327: contextlib.nested is deprecated since Python 2.7. See https://docs.python.org/2/library/contextlib.html#contextlib.nested for more information.'
    if 'with contextlib.nested(' in logical_line or 'with nested(' in logical_line:
        yield (0, msg)