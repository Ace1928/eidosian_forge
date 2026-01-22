import sys
import unittest
from breezy import tests
@unittest.skipIf(sys.version_info[:2] >= (3, 8), "python > 3.8 doesn't allow changing filesystem default encoding")
def test_log_C(self):
    self.disable_missing_extensions_warning()
    out, err = self.run_log_quiet_long(['tree'], env_changes={'LANG': 'C', 'LC_ALL': 'C', 'LC_CTYPE': None, 'LANGUAGE': None, 'PYTHONCOERCECLOCALE': '0', 'PYTHONUTF8': '0'})
    self.assertEqual(b'', err)
    self.assertEqualDiff(b'------------------------------------------------------------\nrevno: 1\ncommitter: ???? Meinel <juju@info.com>\nbranch nick: tree\ntimestamp: Thu 2006-08-24 20:28:17 +0000\nmessage:\n  Unicode ? commit\n', out)