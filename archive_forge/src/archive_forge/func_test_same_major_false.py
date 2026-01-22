from oslotest import base as test_base
from oslo_utils import versionutils
def test_same_major_false(self):
    """With `same_major` set to False, then major version compatibiity
        rule is not enforced, so a current version of 2.0 is deemed to satisfy
        a requirement of 1.0.
        """
    self.assertFalse(versionutils.is_compatible('2.0', '1.0', same_major=False))
    self.assertTrue(versionutils.is_compatible('1.0', '1.0', same_major=False))
    self.assertTrue(versionutils.is_compatible('1.0', '2.0', same_major=False))