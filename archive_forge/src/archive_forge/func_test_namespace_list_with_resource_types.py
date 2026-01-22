from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.tests import functional
def test_namespace_list_with_resource_types(self):
    self.start_server()
    path = '/v2/metadefs/namespaces'
    md_resource = self._create_metadef_resource(path=path, data=GLOBAL_NAMESPACE_DATA)
    self.assertEqual('MyNamespace', md_resource['namespace'])
    resp = self.api_get(path)
    md_resource = resp.json
    self.assertEqual(1, len(md_resource['namespaces']))
    for namespace_obj in md_resource['namespaces']:
        self.assertIn('resource_type_associations', namespace_obj)
    self.set_policy_rules({'get_metadef_namespaces': '@', 'get_metadef_namespace': '@', 'list_metadef_resource_types': '!'})
    resp = self.api_get(path)
    self.assertEqual(403, resp.status_code)
    self.set_policy_rules({'get_metadef_namespaces': '@', 'get_metadef_namespace': '!', 'list_metadef_resource_types': '@'})
    resp = self.api_get(path)
    md_resource = resp.json
    self.assertEqual(0, len(md_resource['namespaces']))
    for namespace_obj in md_resource['namespaces']:
        self.assertNotIn('resource_type_associations', namespace_obj)