import functools
import re
import tokenize
from hacking import core
@skip_ignored_lines
@core.flake8ext
def assert_equal_in(logical_line, filename):
    """Check assertEqual(A in/not in B, True/False) with collection contents

    Check for assertEqual(A in B, True/False), assertEqual(True/False, A in B),
    assertEqual(A not in B, True/False) or assertEqual(True/False, A not in B)
    sentences.

    N324
    """
    res = re_assert_equal_in_end_with_true_or_false.search(logical_line) or re_assert_equal_in_start_with_true_or_false.search(logical_line)
    if res:
        yield (0, 'N324: Use assertIn/NotIn(A, B) rather than assertEqual(A in/not in B, True/False) when checking collection contents.')