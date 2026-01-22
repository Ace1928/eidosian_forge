from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.tests import functional
def test_namespace_delete_tags_basic(self):
    self.start_server()
    path = '/v2/metadefs/namespaces'
    md_resource = self._create_metadef_resource(path, data=GLOBAL_NAMESPACE_DATA)
    namespace = md_resource['namespace']
    self.assertEqual('MyNamespace', namespace)
    self.assertIn('tags', md_resource)
    path = '/v2/metadefs/namespaces/%s/tags' % namespace
    resp = self.api_delete(path)
    self.assertEqual(204, resp.status_code)
    path = '/v2/metadefs/namespaces/%s' % namespace
    resp = self.api_get(path)
    md_resource = resp.json
    self.assertNotIn('tags', md_resource)
    self.assertEqual('MyNamespace', namespace)
    tag_name = 'MyTag'
    path = '/v2/metadefs/namespaces/%s/tags/%s' % (namespace, tag_name)
    md_resource = self._create_metadef_resource(path)
    self.assertEqual('MyTag', md_resource['name'])
    path = '/v2/metadefs/namespaces/%s/tags' % namespace
    self.set_policy_rules({'delete_metadef_namespace': '!', 'get_metadef_namespace': '@'})
    resp = self.api_delete(path)
    self.assertEqual(403, resp.status_code)
    path = '/v2/metadefs/namespaces/%s/tags' % namespace
    self.set_policy_rules({'delete_metadef_namespace': '@', 'delete_metadef_tags': '!', 'get_metadef_namespace': '@'})
    resp = self.api_delete(path)
    self.assertEqual(403, resp.status_code)
    path = '/v2/metadefs/namespaces/non-existing/tags'
    self.set_policy_rules({'delete_metadef_namespace': '@', 'delete_metadef_tags': '@', 'get_metadef_namespace': '@'})
    resp = self.api_delete(path)
    self.assertEqual(404, resp.status_code)
    self.set_policy_rules({'delete_metadef_namespace': '!', 'get_metadef_namespace': '!', 'delete_metadef_tags': '!'})
    path = '/v2/metadefs/namespaces/%s/tags' % namespace
    resp = self.api_delete(path)
    self.assertEqual(404, resp.status_code)
    self.set_policy_rules({'delete_metadef_namespace': '@', 'get_metadef_namespace': '@', 'delete_metadef_tags': '@'})
    self._verify_forbidden_converted_to_not_found(path, 'DELETE')