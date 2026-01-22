import sys
from io import StringIO
from unittest import TestCase
from IPython.testing import tools as tt
from IPython.core.magic import (
def test_cpaste():
    """Test cpaste magic"""

    def runf():
        """Marker function: sets a flag when executed.
        """
        ip.user_ns['code_ran'] = True
        return 'runf'
    tests = {'pass': ['runf()', 'In [1]: runf()', 'In [1]: if 1:\n   ...:     runf()', '> > > runf()', '>>> runf()', '   >>> runf()'], 'fail': ['1 + runf()', '++ runf()']}
    ip.user_ns['runf'] = runf
    for code in tests['pass']:
        check_cpaste(code)
    for code in tests['fail']:
        check_cpaste(code, should_fail=True)