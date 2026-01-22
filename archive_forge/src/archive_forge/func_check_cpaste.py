import sys
from io import StringIO
from unittest import TestCase
from IPython.testing import tools as tt
from IPython.core.magic import (
def check_cpaste(code, should_fail=False):
    """Execute code via 'cpaste' and ensure it was executed, unless
    should_fail is set.
    """
    ip.user_ns['code_ran'] = False
    src = StringIO()
    src.write(code)
    src.write('\n--\n')
    src.seek(0)
    stdin_save = sys.stdin
    sys.stdin = src
    try:
        context = tt.AssertPrints if should_fail else tt.AssertNotPrints
        with context('Traceback (most recent call last)'):
            ip.run_line_magic('cpaste', '')
        if not should_fail:
            assert ip.user_ns['code_ran'], '%r failed' % code
    finally:
        sys.stdin = stdin_save