from oslotest import base as test_base
from oslo_utils import versionutils
def test_convert_version_to_tuple(self):
    self.assertEqual((6, 7, 0), versionutils.convert_version_to_tuple('6.7.0'))
    self.assertEqual((6, 7, 0), versionutils.convert_version_to_tuple('6.7.0a1'))
    self.assertEqual((6, 7, 0), versionutils.convert_version_to_tuple('6.7.0alpha1'))
    self.assertEqual((6, 7, 0), versionutils.convert_version_to_tuple('6.7.0b1'))
    self.assertEqual((6, 7, 0), versionutils.convert_version_to_tuple('6.7.0beta1'))
    self.assertEqual((6, 7, 0), versionutils.convert_version_to_tuple('6.7.0rc1'))