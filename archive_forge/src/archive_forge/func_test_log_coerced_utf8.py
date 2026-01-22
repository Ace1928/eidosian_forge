import sys
import unittest
from breezy import tests
@unittest.skip('encoding when LANG=C is currently borked')
def test_log_coerced_utf8(self):
    self.disable_missing_extensions_warning()
    out, err = self.run_log_quiet_long(['tree'], env_changes={'LANG': 'C', 'LC_ALL': 'C', 'LC_CTYPE': None, 'LANGUAGE': None})
    self.assertEqual(b'', err)
    self.assertEqualDiff(b'------------------------------------------------------------\nrevno: 1\ncommitter: \xd8\xac\xd9\x88\xd8\xac\xd9\x88 Meinel <juju@info.com>\nbranch nick: tree\ntimestamp: Thu 2006-08-24 20:28:17 +0000\nmessage:\n  Unicode \xc2\xb5 commit\n', out)