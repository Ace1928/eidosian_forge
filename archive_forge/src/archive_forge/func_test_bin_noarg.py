import sys
from tests.base import BaseTestCase
from pyasn1.compat import binary
def test_bin_noarg(self):
    try:
        binary.bin()
    except TypeError:
        pass
    except:
        assert 0, 'bin() tolerates no arguments'