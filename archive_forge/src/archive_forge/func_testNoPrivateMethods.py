import sys
import fire
from fire import testutils
import mock
def testNoPrivateMethods(self):
    self.assertTrue(hasattr(fire, 'Fire'))
    self.assertFalse(hasattr(fire, '_Fire'))