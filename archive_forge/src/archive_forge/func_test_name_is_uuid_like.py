import uuid
from oslotest import base as test_base
from oslo_utils import uuidutils
def test_name_is_uuid_like(self):
    self.assertFalse(uuidutils.is_uuid_like('zhongyueluo'))