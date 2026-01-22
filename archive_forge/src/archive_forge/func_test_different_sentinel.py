import datetime
from oslotest import base as test_base
from oslo_utils import fixture
from oslo_utils.fixture import keystoneidsentinel as keystids
from oslo_utils.fixture import uuidsentinel as uuids
from oslo_utils import timeutils
from oslo_utils import uuidutils
def test_different_sentinel(self):
    uuid1 = uuids.foobar
    uuid2 = uuids.barfoo
    self.assertNotEqual(uuid1, uuid2)
    keystid1 = keystids.foobar
    keystid2 = keystids.barfoo
    self.assertNotEqual(keystid1, keystid2)