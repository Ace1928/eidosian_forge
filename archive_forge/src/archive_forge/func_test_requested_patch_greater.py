from oslotest import base as test_base
from oslo_utils import versionutils
def test_requested_patch_greater(self):
    self.assertFalse(versionutils.is_compatible('1.0.1', '1.0.0'))