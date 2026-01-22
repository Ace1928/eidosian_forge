import testtools
from testtools.matchers import HasLength
from ironicclient.tests.unit import utils
from ironicclient.v1 import conductor
def test_conductor_list_sort_dir(self):
    self.api = utils.FakeAPI(fake_responses_sorting)
    self.mgr = conductor.ConductorManager(self.api)
    conductors = self.mgr.list(sort_dir='desc')
    expect = [('GET', '/v1/conductors/?sort_dir=desc', {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(2, len(conductors))