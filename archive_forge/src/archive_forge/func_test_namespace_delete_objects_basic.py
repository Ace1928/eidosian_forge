from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.tests import functional
def test_namespace_delete_objects_basic(self):
    self.start_server()
    path = '/v2/metadefs/namespaces'
    md_resource = self._create_metadef_resource(path, data=GLOBAL_NAMESPACE_DATA)
    self.assertEqual('MyNamespace', md_resource['namespace'])
    self.assertIn('objects', md_resource)
    path = '/v2/metadefs/namespaces/%s/objects' % md_resource['namespace']
    resp = self.api_delete(path)
    self.assertEqual(204, resp.status_code)
    path = '/v2/metadefs/namespaces/%s' % md_resource['namespace']
    resp = self.api_get(path)
    md_resource = resp.json
    self.assertNotIn('objects', md_resource)
    self.assertEqual('MyNamespace', md_resource['namespace'])
    path = '/v2/metadefs/namespaces/%s/objects' % md_resource['namespace']
    data = {'name': 'MyObject', 'description': 'My object for My namespace', 'properties': {'test_property': {'title': 'test_property', 'description': 'Test property for My object', 'type': 'string'}}}
    md_object = self._create_metadef_resource(path, data=data)
    self.assertEqual('MyObject', md_object['name'])
    path = '/v2/metadefs/namespaces/%s/objects' % md_resource['namespace']
    self.set_policy_rules({'delete_metadef_namespace': '!', 'get_metadef_namespace': '@'})
    resp = self.api_delete(path)
    self.assertEqual(403, resp.status_code)
    path = '/v2/metadefs/namespaces/non-existing/objects'
    self.set_policy_rules({'delete_metadef_namespace': '@', 'get_metaded_namespace': '@'})
    resp = self.api_delete(path)
    self.assertEqual(404, resp.status_code)
    self.set_policy_rules({'delete_metadef_namespace': '!', 'get_metadef_namespace': '!'})
    path = '/v2/metadefs/namespaces/%s/objects' % md_resource['namespace']
    resp = self.api_delete(path)
    self.assertEqual(404, resp.status_code)
    self.set_policy_rules({'delete_metadef_namespace': '@', 'get_metadef_namespace': '@'})
    self._verify_forbidden_converted_to_not_found(path, 'DELETE')