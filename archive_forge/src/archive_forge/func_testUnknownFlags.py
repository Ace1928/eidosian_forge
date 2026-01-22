import sys
from tests.base import BaseTestCase
from pyasn1 import debug
from pyasn1 import error
def testUnknownFlags(self):
    try:
        debug.setLogger(debug.Debug('all', 'unknown', loggerName='xxx'))
    except error.PyAsn1Error:
        debug.setLogger(0)
        return
    else:
        debug.setLogger(0)
        assert 0, 'unknown debug flag tolerated'