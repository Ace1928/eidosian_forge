from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.tests import functional
def test_tag_update_basic(self):
    self.start_server()
    self.load_data(create_tags=True)
    namespace = NAME_SPACE1['namespace']
    path = '/v2/metadefs/namespaces/%s/tags/%s' % (namespace, TAG1['name'])
    data = {'name': 'MyTagUpdated'}
    resp = self.api_put(path, json=data)
    md_resource = resp.json
    self.assertEqual('MyTagUpdated', md_resource['name'])
    self.set_policy_rules({'modify_metadef_tag': '!', 'get_metadef_namespace': ''})
    path = '/v2/metadefs/namespaces/%s/tags/%s' % (namespace, TAG2['name'])
    data = {'name': 'MySecondTagUpdated'}
    resp = self.api_put(path, json=data)
    self.assertEqual(403, resp.status_code)
    self.set_policy_rules({'modify_metadef_tag': '!', 'get_metadef_namespace': '!'})
    resp = self.api_put(path, json=data)
    self.assertEqual(404, resp.status_code)
    self.set_policy_rules({'get_metadef_tag': '', 'get_metadef_namespace': ''})
    self._verify_forbidden_converted_to_not_found(path, 'PUT', json=data)