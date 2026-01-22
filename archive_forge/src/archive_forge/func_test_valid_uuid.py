import testtools
import uuid
import barbicanclient
from barbicanclient import base
from barbicanclient import version
def test_valid_uuid(self):
    secret_uuid = uuid.uuid4()
    self.assertEqual(secret_uuid, base.validate_ref_and_return_uuid(str(secret_uuid), 'Thing'))