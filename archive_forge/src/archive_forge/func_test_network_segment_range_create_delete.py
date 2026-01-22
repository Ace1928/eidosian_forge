import uuid
from openstackclient.tests.functional.network.v2 import common
def test_network_segment_range_create_delete(self):
    project_id = self.openstack('project create ' + self.PROJECT_NAME, parse_output=True)['id']
    name = uuid.uuid4().hex
    json_output = self.openstack(' network segment range create ' + '--private ' + '--project ' + self.PROJECT_NAME + ' ' + '--network-type vxlan ' + '--minimum 2005 ' + '--maximum 2009 ' + name, parse_output=True)
    self.assertEqual(name, json_output['name'])
    self.assertEqual(project_id, json_output['project_id'])
    raw_output = self.openstack('network segment range delete ' + name)
    self.assertOutput('', raw_output)
    raw_output_project = self.openstack('project delete ' + self.PROJECT_NAME)
    self.assertEqual('', raw_output_project)