from numpy.core._machar import MachAr
import numpy.core.numerictypes as ntypes
from numpy import errstate, array
def test_underlow(self):
    with errstate(all='raise'):
        try:
            self._run_machar_highprec()
        except FloatingPointError as e:
            msg = 'Caught %s exception, should not have been raised.' % e
            raise AssertionError(msg)