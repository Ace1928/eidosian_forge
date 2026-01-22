import testtools
import uuid
import barbicanclient
from barbicanclient import base
from barbicanclient import version
def test_invalid_uuid(self):
    ref = 'http://localhost/not_a_uuid'
    self.assertRaises(ValueError, base.validate_ref_and_return_uuid, ref, 'Thing')