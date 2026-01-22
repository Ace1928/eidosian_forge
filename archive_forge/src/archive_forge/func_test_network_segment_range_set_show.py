import uuid
from openstackclient.tests.functional.network.v2 import common
def test_network_segment_range_set_show(self):
    project_id = self.openstack('project create ' + self.PROJECT_NAME, parse_output=True)['id']
    name = uuid.uuid4().hex
    json_output = self.openstack(' network segment range create ' + '--private ' + '--project ' + self.PROJECT_NAME + ' ' + '--network-type geneve ' + '--minimum 2021 ' + '--maximum 2025 ' + name, parse_output=True)
    self.addCleanup(self.openstack, 'network segment range delete ' + name)
    self.assertEqual(name, json_output['name'])
    self.assertEqual(project_id, json_output['project_id'])
    new_minimum = 2020
    new_maximum = 2029
    cmd_output = self.openstack('network segment range set --minimum {min} --maximum {max} {name}'.format(min=new_minimum, max=new_maximum, name=name))
    self.assertOutput('', cmd_output)
    json_output = self.openstack('network segment range show ' + name, parse_output=True)
    self.assertEqual(new_minimum, json_output['minimum'])
    self.assertEqual(new_maximum, json_output['maximum'])
    raw_output_project = self.openstack('project delete ' + self.PROJECT_NAME)
    self.assertEqual('', raw_output_project)