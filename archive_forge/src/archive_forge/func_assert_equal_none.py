import functools
import re
import tokenize
from hacking import core
@skip_ignored_lines
@core.flake8ext
def assert_equal_none(logical_line, filename):
    """Check for assertEqual(A, None) or assertEqual(None, A) sentences

    N322
    """
    res = re_assert_equal_start_with_none.search(logical_line) or re_assert_equal_end_with_none.search(logical_line)
    if res:
        yield (0, 'N322 assertEqual(A, None) or assertEqual(None, A) sentences not allowed, you should use assertIsNone(A) instead.')