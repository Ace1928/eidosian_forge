import testtools
from oslotest import base
from octaviaclient.hacking import checks
def test_no_xrange(self):
    self.assertEqual(1, len(list(checks.no_xrange('xrange(45)'))))
    self.assertEqual(0, len(list(checks.no_xrange('range(45)'))))