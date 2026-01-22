from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.tests import functional
def test_namespace_get_basic(self):
    self.start_server()
    path = '/v2/metadefs/namespaces'
    md_resource = self._create_metadef_resource(path=path, data=GLOBAL_NAMESPACE_DATA)
    self.assertEqual('MyNamespace', md_resource['namespace'])
    path = '/v2/metadefs/namespaces/%s' % md_resource['namespace']
    resp = self.api_get(path)
    md_resource = resp.json
    self.assertEqual('MyNamespace', md_resource['namespace'])
    self.assertIn('objects', md_resource)
    self.assertIn('resource_type_associations', md_resource)
    self.assertIn('tags', md_resource)
    self.assertIn('properties', md_resource)
    self.set_policy_rules({'get_metadef_namespace': '!'})
    path = '/v2/metadefs/namespaces/%s' % md_resource['namespace']
    resp = self.api_get(path)
    self.assertEqual(404, resp.status_code)
    self.set_policy_rules({'get_metadef_namespace': '@'})
    self._verify_forbidden_converted_to_not_found(path, 'GET')
    self.set_policy_rules({'get_metadef_objects': '!', 'get_metadef_namespace': '@', 'list_metadef_resource_types': '@', 'get_metadef_properties': '@', 'get_metadef_tags': '@'})
    path = '/v2/metadefs/namespaces/%s' % md_resource['namespace']
    resp = self.api_get(path)
    self.assertEqual(403, resp.status_code)
    self.set_policy_rules({'get_metadef_objects': '@', 'get_metadef_namespace': '@', 'list_metadef_resource_types': '!', 'get_metadef_properties': '@', 'get_metadef_tags': '@'})
    path = '/v2/metadefs/namespaces/%s' % md_resource['namespace']
    resp = self.api_get(path)
    self.assertEqual(403, resp.status_code)
    self.set_policy_rules({'get_metadef_objects': '@', 'get_metadef_namespace': '@', 'list_metadef_resource_types': '@', 'get_metadef_properties': '!', 'get_metadef_tags': '@'})
    path = '/v2/metadefs/namespaces/%s' % md_resource['namespace']
    resp = self.api_get(path)
    self.assertEqual(403, resp.status_code)
    self.set_policy_rules({'get_metadef_objects': '@', 'get_metadef_namespace': '@', 'list_metadef_resource_types': '@', 'get_metadef_properties': '@', 'get_metadef_tags': '!'})
    path = '/v2/metadefs/namespaces/%s' % md_resource['namespace']
    resp = self.api_get(path)
    self.assertEqual(403, resp.status_code)