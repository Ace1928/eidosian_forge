from saharaclient.api import node_group_templates as ng
from saharaclient.tests.unit import base
from oslo_serialization import jsonutils as json
def test_node_group_template_get(self):
    url = self.URL + '/node-group-templates/id'
    self.responses.get(url, json={'node_group_template': self.body})
    resp = self.client.node_group_templates.get('id')
    self.assertEqual(url, self.responses.last_request.url)
    self.assertIsInstance(resp, ng.NodeGroupTemplate)
    self.assertFields(self.body, resp)