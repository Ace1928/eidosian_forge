from oslotest import base as test_base
from oslo_utils import versionutils
def test_current_patch_not_present_less_than(self):
    self.assertFalse(versionutils.is_compatible('1.0.1', '1.0'))