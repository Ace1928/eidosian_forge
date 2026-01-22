import testtools
from oslotest import base
from octavia_lib.hacking import checks
def test_line_continuation_no_backslash(self):
    results = list(checks.check_line_continuation_no_backslash('', [(1, 'import', (2, 0), (2, 6), 'import \\\n'), (1, 'os', (3, 4), (3, 6), '    os\n')]))
    self.assertEqual(1, len(results))
    self.assertEqual((2, 7), results[0][0])