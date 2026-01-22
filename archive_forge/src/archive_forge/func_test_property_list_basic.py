from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.tests import functional
def test_property_list_basic(self):
    self.start_server()
    self.load_data(create_properties=True)
    namespace = NAME_SPACE1['namespace']
    path = '/v2/metadefs/namespaces/%s/properties' % namespace
    resp = self.api_get(path)
    md_resource = resp.json
    self.assertEqual(2, len(md_resource['properties']))
    self.set_policy_rules({'get_metadef_properties': '!', 'get_metadef_namespace': ''})
    resp = self.api_get(path)
    self.assertEqual(403, resp.status_code)
    self.set_policy_rules({'get_metadef_properties': '!', 'get_metadef_namespace': '!'})
    resp = self.api_get(path)
    self.assertEqual(404, resp.status_code)
    self.set_policy_rules({'get_metadef_properties': '', 'get_metadef_namespace': ''})
    self._verify_forbidden_converted_to_not_found(path, 'GET')