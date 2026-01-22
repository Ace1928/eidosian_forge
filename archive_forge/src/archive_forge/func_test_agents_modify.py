from novaclient.tests.unit.fixture_data import agents as data
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import agents
def test_agents_modify(self):
    ag = self.cs.agents.update('1', '8.0', '/yyy/yyyy/yyyy', 'add6bb58e139be103324d04d82d8f546')
    self.assert_request_id(ag, fakes.FAKE_REQUEST_ID_LIST)
    body = self._build_example_update_body()
    self.assert_called('PUT', '/os-agents/1', body)
    self.assertEqual(1, ag.id)