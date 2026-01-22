from glance.hacking import checks
from glance.tests import utils
def test_assert_equal_type(self):
    self.assertEqual(1, len(list(checks.assert_equal_type("self.assertEqual(type(also['QuicAssist']), list)"))))
    self.assertEqual(0, len(list(checks.assert_equal_type('self.assertTrue()'))))