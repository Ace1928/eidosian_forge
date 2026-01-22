import os
import unittest
from IPython.testing import decorators as dec
from IPython.testing import tools as tt
def test_main_path(self):
    """Test with only stdout results.
        """
    self.mktmp("print('A')\nprint('B')\n")
    out = 'A\nB'
    tt.ipexec_validate(self.fname, out)