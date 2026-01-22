import datetime
from oslotest import base as test_base
from oslo_utils import fixture
from oslo_utils.fixture import keystoneidsentinel as keystids
from oslo_utils.fixture import uuidsentinel as uuids
from oslo_utils import timeutils
from oslo_utils import uuidutils
def test_set_time_override_using_default(self):
    self.assertIsNone(timeutils.utcnow.override_time)
    with fixture.TimeFixture():
        self.assertIsNotNone(timeutils.utcnow.override_time)
    self.assertIsNone(timeutils.utcnow.override_time)