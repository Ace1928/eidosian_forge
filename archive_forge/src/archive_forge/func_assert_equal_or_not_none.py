import re
from hacking import core
@core.flake8ext
def assert_equal_or_not_none(logical_line):
    """Check for assertEqual(A, None) or assertEqual(None, A) sentences,

    assertNotEqual(A, None) or assertNotEqual(None, A) sentences

    O318
    """
    msg = 'O318: assertEqual/assertNotEqual(A, None) or assertEqual/assertNotEqual(None, A) sentences not allowed'
    res = assert_equal_start_with_none_re.match(logical_line) or assert_equal_end_with_none_re.match(logical_line) or assert_not_equal_start_with_none_re.match(logical_line) or assert_not_equal_end_with_none_re.match(logical_line)
    if res:
        yield (0, msg)