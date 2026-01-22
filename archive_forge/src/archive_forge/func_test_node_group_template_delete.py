from saharaclient.api import node_group_templates as ng
from saharaclient.tests.unit import base
from oslo_serialization import jsonutils as json
def test_node_group_template_delete(self):
    url = self.URL + '/node-group-templates/id'
    self.responses.delete(url, status_code=204)
    self.client.node_group_templates.delete('id')
    self.assertEqual(url, self.responses.last_request.url)