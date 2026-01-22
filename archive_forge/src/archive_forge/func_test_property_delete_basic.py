from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.tests import functional
def test_property_delete_basic(self):
    self.start_server()
    self.load_data(create_properties=True)
    path = '/v2/metadefs/namespaces/%s/properties/%s' % (NAME_SPACE1['namespace'], PROPERTY1['name'])
    resp = self.api_delete(path)
    self.assertEqual(204, resp.status_code)
    path = '/v2/metadefs/namespaces/%s/properties/%s' % (NAME_SPACE1['namespace'], PROPERTY1['name'])
    resp = self.api_get(path)
    self.assertEqual(404, resp.status_code)
    path = '/v2/metadefs/namespaces/%s/properties/%s' % (NAME_SPACE1['namespace'], PROPERTY2['name'])
    self.set_policy_rules({'remove_metadef_property': '!', 'get_metadef_namespace': ''})
    resp = self.api_delete(path)
    self.assertEqual(403, resp.status_code)
    self.set_policy_rules({'remove_metadef_property': '!', 'get_metadef_namespace': '!'})
    resp = self.api_delete(path)
    self.assertEqual(404, resp.status_code)
    self.set_policy_rules({'remove_metadef_property': '', 'get_metadef_namespace': ''})
    self._verify_forbidden_converted_to_not_found(path, 'DELETE')