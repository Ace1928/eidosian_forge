import os
import unittest
from IPython.testing import decorators as dec
from IPython.testing import tools as tt
def test_main_path2(self):
    """Test with only stdout results, expecting windows line endings.
        """
    self.mktmp("print('A')\nprint('B')\n")
    out = 'A\r\nB'
    tt.ipexec_validate(self.fname, out)