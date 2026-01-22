import datetime
from oslotest import base as test_base
from oslo_utils import fixture
from oslo_utils.fixture import keystoneidsentinel as keystids
from oslo_utils.fixture import uuidsentinel as uuids
from oslo_utils import timeutils
from oslo_utils import uuidutils
def test_returns_uuid(self):
    self.assertTrue(uuidutils.is_uuid_like(uuids.foo))
    self.assertTrue(uuidutils.is_uuid_like(keystids.foo))