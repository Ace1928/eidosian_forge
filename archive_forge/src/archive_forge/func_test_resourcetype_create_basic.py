from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.tests import functional
def test_resourcetype_create_basic(self):
    self.start_server()
    self.load_data()
    namespace = NAME_SPACE1['namespace']
    path = '/v2/metadefs/namespaces/%s/resource_types' % namespace
    md_resource = self._create_metadef_resource(path=path, data=RESOURCETYPE_1)
    self.assertEqual('MyResourceType', md_resource['name'])
    self.set_policy_rules({'add_metadef_resource_type_association': '!', 'get_metadef_namespace': '@'})
    resp = self.api_post(path, json=RESOURCETYPE_2)
    self.assertEqual(403, resp.status_code)
    self.set_policy_rules({'add_metadef_resource_type_association': '!', 'get_metadef_namespace': '!'})
    resp = self.api_post(path, json=RESOURCETYPE_2)
    self.assertEqual(404, resp.status_code)
    self.set_policy_rules({'add_metadef_resource_type_association': '@', 'get_metadef_namespace': '@'})
    self._verify_forbidden_converted_to_not_found(path, 'POST', json=RESOURCETYPE_2)