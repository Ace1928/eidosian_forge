import testtools
from testtools.matchers import HasLength
from ironicclient.tests.unit import utils
from ironicclient.v1 import conductor
def test_conductor_list_pagination_no_limit(self):
    self.api = utils.FakeAPI(fake_responses_pagination)
    self.mgr = conductor.ConductorManager(self.api)
    conductors = self.mgr.list(limit=0)
    expect = [('GET', '/v1/conductors', {}, None), ('GET', '/v1/conductors/?limit=1', {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(2, len(conductors))