from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.tests import functional
def test_namespace_update_basic(self):
    self.start_server()
    path = '/v2/metadefs/namespaces'
    md_resource = self._create_metadef_resource(path=path, data=NAME_SPACE1)
    self.assertEqual('MyNamespace', md_resource['namespace'])
    self.assertEqual('private', md_resource['visibility'])
    path = '/v2/metadefs/namespaces/%s' % md_resource['namespace']
    data = {'visibility': 'public', 'namespace': md_resource['namespace']}
    resp = self.api_put(path, json=data)
    md_resource = resp.json
    self.assertEqual('MyNamespace', md_resource['namespace'])
    self.assertEqual('public', md_resource['visibility'])
    self.set_policy_rules({'modify_metadef_namespace': '!', 'get_metadef_namespace': '@'})
    resp = self.api_put(path, json=data)
    self.assertEqual(403, resp.status_code)
    self.set_policy_rules({'modify_metadef_namespace': '@', 'get_metadef_namespace': '@'})
    path = '/v2/metadefs/namespaces/non-existing'
    resp = self.api_put(path, json=data)
    self.assertEqual(404, resp.status_code)
    self.set_policy_rules({'modify_metadef_namespace': '!', 'get_metadef_namespace': '!'})
    path = '/v2/metadefs/namespaces/%s' % md_resource['namespace']
    resp = self.api_put(path, json=data)
    self.assertEqual(404, resp.status_code)
    self.set_policy_rules({'modify_metadef_namespace': '@', 'get_metadef_namespace': '@'})
    path = '/v2/metadefs/namespaces/%s' % md_resource['namespace']
    data = {'visibility': 'private', 'namespace': md_resource['namespace']}
    resp = self.api_put(path, json=data)
    md_resource = resp.json
    self.assertEqual('MyNamespace', md_resource['namespace'])
    self.assertEqual('private', md_resource['visibility'])
    self._verify_forbidden_converted_to_not_found(path, 'PUT', json=data)