from saharaclient.api import node_group_templates as ng
from saharaclient.tests.unit import base
from oslo_serialization import jsonutils as json
def test_update_node_group_template(self):
    url = self.URL + '/node-group-templates'
    self.responses.post(url, status_code=202, json={'node_group_template': self.body})
    resp = self.client.node_group_templates.create(**self.body)
    update_url = self.URL + '/node-group-templates/id'
    self.responses.put(update_url, status_code=202, json=self.update_json)
    updated = self.client.node_group_templates.update('id', resp.name, resp.plugin_name, resp.hadoop_version, resp.flavor_id, description=getattr(resp, 'description', None), volumes_per_node=getattr(resp, 'volumes_per_node', None), node_configs=getattr(resp, 'node_configs', None), floating_ip_pool=getattr(resp, 'floating_ip_pool', None), security_groups=getattr(resp, 'security_groups', None), auto_security_group=getattr(resp, 'auto_security_group', None), availability_zone=getattr(resp, 'availability_zone', None), volumes_availability_zone=getattr(resp, 'volumes_availability_zone', None), volume_type=getattr(resp, 'volume_type', None), image_id=getattr(resp, 'image_id', None), is_proxy_gateway=getattr(resp, 'is_proxy_gateway', None), volume_local_to_instance=getattr(resp, 'volume_local_to_instance', None), use_autoconfig=False)
    self.assertIsInstance(updated, ng.NodeGroupTemplate)
    self.assertFields(self.update_json['node_group_template'], updated)
    self.client.node_group_templates.update('id')
    self.assertEqual(update_url, self.responses.last_request.url)
    self.assertEqual({}, json.loads(self.responses.last_request.body))
    unset_json = {'auto_security_group': None, 'availability_zone': None, 'description': None, 'flavor_id': None, 'floating_ip_pool': None, 'hadoop_version': None, 'image_id': None, 'is_protected': None, 'is_proxy_gateway': None, 'is_public': None, 'name': None, 'node_configs': None, 'node_processes': None, 'plugin_name': None, 'security_groups': None, 'shares': None, 'use_autoconfig': None, 'volume_local_to_instance': None, 'volume_mount_prefix': None, 'volume_type': None, 'volumes_availability_zone': None, 'volumes_per_node': None, 'volumes_size': None}
    self.client.node_group_templates.update('id', **unset_json)
    self.assertEqual(update_url, self.responses.last_request.url)
    self.assertEqual(unset_json, json.loads(self.responses.last_request.body))