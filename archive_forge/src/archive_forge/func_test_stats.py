from ...tests import TestCaseWithTransport
def test_stats(self):
    out, err = self.run_bzr('stats')
    self.assertEqual(out, '   3 Fero <fero@example.com>\n     Other names:\n        2 Fero\n        1 Ferko\n   1 Vinco <vinco@example.com>\n   1 Jano <jano@example.com>\n')