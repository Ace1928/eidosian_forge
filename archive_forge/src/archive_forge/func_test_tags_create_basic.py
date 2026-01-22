from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.tests import functional
def test_tags_create_basic(self):
    self.start_server()
    self.load_data()
    namespace = NAME_SPACE1['namespace']
    path = '/v2/metadefs/namespaces/%s/tags' % namespace
    data = {'tags': [TAG1, TAG2]}
    md_resource = self._create_metadef_resource(path=path, data=data)
    self.assertEqual(2, len(md_resource['tags']))
    self.set_policy_rules({'add_metadef_tags': '!', 'get_metadef_namespace': ''})
    path = '/v2/metadefs/namespaces/%s/tags' % namespace
    data = {'tags': [{'name': 'sampe-tag-1'}, {'name': 'sampe-tag-2'}]}
    resp = self.api_post(path, json=data)
    self.assertEqual(403, resp.status_code)
    self.set_policy_rules({'add_metadef_tags': '!', 'get_metadef_namespace': '!'})
    resp = self.api_post(path, json=data)
    self.assertEqual(404, resp.status_code)
    self.set_policy_rules({'add_metadef_tags': '', 'get_metadef_namespace': ''})
    self._verify_forbidden_converted_to_not_found(path, 'POST', json=data)