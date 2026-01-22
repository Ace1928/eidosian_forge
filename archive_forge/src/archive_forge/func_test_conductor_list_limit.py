import testtools
from testtools.matchers import HasLength
from ironicclient.tests.unit import utils
from ironicclient.v1 import conductor
def test_conductor_list_limit(self):
    self.api = utils.FakeAPI(fake_responses_pagination)
    self.mgr = conductor.ConductorManager(self.api)
    conductors = self.mgr.list(limit=1)
    expect = [('GET', '/v1/conductors/?limit=1', {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertThat(conductors, HasLength(1))