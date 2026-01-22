from saharaclient.api import cluster_templates as ct
from saharaclient.tests.unit import base
from oslo_serialization import jsonutils as json
def test_cluster_template_update(self):
    url = self.URL + '/cluster-templates'
    self.responses.post(url, status_code=202, json={'cluster_template': self.body})
    resp = self.client.cluster_templates.create(**self.body)
    update_url = self.URL + '/cluster-templates/id'
    self.responses.put(update_url, status_code=202, json=self.update_json)
    updated = self.client.cluster_templates.update('id', resp.name, resp.plugin_name, resp.hadoop_version, description=getattr(resp, 'description', None), cluster_configs=getattr(resp, 'cluster_configs', None), node_groups=getattr(resp, 'node_groups', None), anti_affinity=getattr(resp, 'anti_affinity', None), net_id=getattr(resp, 'neutron_management_network', None), default_image_id=getattr(resp, 'default_image_id', None), use_autoconfig=True, domain_name=getattr(resp, 'domain_name', None))
    self.assertIsInstance(updated, ct.ClusterTemplate)
    self.assertFields(self.update_json['cluster_template'], updated)
    self.client.cluster_templates.update('id')
    self.assertEqual(update_url, self.responses.last_request.url)
    self.assertEqual({}, json.loads(self.responses.last_request.body))
    unset_json = {'anti_affinity': None, 'cluster_configs': None, 'default_image_id': None, 'description': None, 'hadoop_version': None, 'is_protected': None, 'is_public': None, 'name': None, 'net_id': None, 'node_groups': None, 'plugin_name': None, 'shares': None, 'use_autoconfig': None, 'domain_name': None}
    req_json = unset_json.copy()
    req_json['neutron_management_network'] = req_json.pop('net_id')
    self.client.cluster_templates.update('id', **unset_json)
    self.assertEqual(update_url, self.responses.last_request.url)
    self.assertEqual(req_json, json.loads(self.responses.last_request.body))