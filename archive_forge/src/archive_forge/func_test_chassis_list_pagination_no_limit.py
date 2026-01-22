import copy
import testtools
from testtools.matchers import HasLength
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.chassis
def test_chassis_list_pagination_no_limit(self):
    self.api = utils.FakeAPI(fake_responses_pagination)
    self.mgr = ironicclient.v1.chassis.ChassisManager(self.api)
    chassis = self.mgr.list(limit=0)
    expect = [('GET', '/v1/chassis', {}, None), ('GET', '/v1/chassis/?limit=1', {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertThat(chassis, HasLength(2))