from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3.volume_encryption_types import VolumeEncryptionType
def test_get_no_encryption(self):
    """
        Unit test for VolumeEncryptionTypesManager.get

        Verify that a request on a volume type with no associated encryption
        type information returns a VolumeEncryptionType with no attributes.
        """
    encryption_type = cs.volume_encryption_types.get(2)
    self.assertIsInstance(encryption_type, VolumeEncryptionType)
    self.assertFalse(hasattr(encryption_type, 'id'), 'encryption type has an id')
    self._assert_request_id(encryption_type)