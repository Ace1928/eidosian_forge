from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.tests import functional
def test_property_get_basic(self):
    self.start_server()
    self.load_data(create_properties=True)
    path = '/v2/metadefs/namespaces/%s/properties/%s' % (NAME_SPACE1['namespace'], PROPERTY1['name'])
    resp = self.api_get(path)
    md_resource = resp.json
    self.assertEqual(PROPERTY1['name'], md_resource['name'])
    self.set_policy_rules({'get_metadef_property': '!', 'get_metadef_namespace': '', 'get_metadef_resource_type': ''})
    resp = self.api_get(path)
    self.assertEqual(403, resp.status_code)
    self.set_policy_rules({'get_metadef_property': '', 'get_metadef_namespace': '', 'get_metadef_resource_type': '!'})
    url_path = "%s?resource_type='abcd'" % path
    resp = self.api_get(url_path)
    self.assertEqual(403, resp.status_code)
    self.set_policy_rules({'get_metadef_property': '!', 'get_metadef_namespace': '!', 'get_metadef_resource_type': '!'})
    resp = self.api_get(path)
    self.assertEqual(404, resp.status_code)
    self.set_policy_rules({'get_metadef_property': '', 'get_metadef_namespace': '', 'get_metadef_resource_type': ''})
    self._verify_forbidden_converted_to_not_found(path, 'GET')