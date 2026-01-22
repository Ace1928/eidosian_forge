from unittest import mock
from openstack.network.v2 import agent
from openstack.tests.unit import base
def test_add_router_to_agent(self):
    sot = agent.Agent(**EXAMPLE)
    response = mock.Mock()
    response.body = {'router_id': '1'}
    response.json = mock.Mock(return_value=response.body)
    sess = mock.Mock()
    sess.post = mock.Mock(return_value=response)
    router_id = '1'
    self.assertEqual(response.body, sot.add_router_to_agent(sess, router_id))
    body = {'router_id': router_id}
    url = 'agents/IDENTIFIER/l3-routers'
    sess.post.assert_called_with(url, json=body)