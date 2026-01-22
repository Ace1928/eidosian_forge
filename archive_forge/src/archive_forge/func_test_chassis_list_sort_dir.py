import copy
import testtools
from testtools.matchers import HasLength
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.chassis
def test_chassis_list_sort_dir(self):
    self.api = utils.FakeAPI(fake_responses_sorting)
    self.mgr = ironicclient.v1.chassis.ChassisManager(self.api)
    chassis = self.mgr.list(sort_dir='desc')
    expect = [('GET', '/v1/chassis/?sort_dir=desc', {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertThat(chassis, HasLength(1))