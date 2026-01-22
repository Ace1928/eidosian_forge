from glance.hacking import checks
from glance.tests import utils
def test_assert_equal_none(self):
    self.assertEqual(1, len(list(checks.assert_equal_none('self.assertEqual(A, None)'))))
    self.assertEqual(1, len(list(checks.assert_equal_none('self.assertEqual(None, A)'))))
    self.assertEqual(0, len(list(checks.assert_equal_none('self.assertIsNone()'))))