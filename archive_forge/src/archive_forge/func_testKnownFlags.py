import sys
from tests.base import BaseTestCase
from pyasn1 import debug
from pyasn1 import error
def testKnownFlags(self):
    debug.setLogger(0)
    debug.setLogger(debug.Debug('all', 'encoder', 'decoder'))
    debug.setLogger(0)