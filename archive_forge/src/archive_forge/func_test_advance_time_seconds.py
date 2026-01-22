import datetime
from oslotest import base as test_base
from oslo_utils import fixture
from oslo_utils.fixture import keystoneidsentinel as keystids
from oslo_utils.fixture import uuidsentinel as uuids
from oslo_utils import timeutils
from oslo_utils import uuidutils
def test_advance_time_seconds(self):
    new_time = datetime.datetime(2015, 1, 2, 3, 4, 6, 7)
    time_fixture = self.useFixture(fixture.TimeFixture(new_time))
    time_fixture.advance_time_seconds(2)
    expected_time = datetime.datetime(2015, 1, 2, 3, 4, 8, 7)
    self.assertEqual(expected_time, timeutils.utcnow())