import http.client as http
from oslo_serialization import jsonutils
import requests
from glance.tests.functional.v2 import metadef_base
def test_role_base_metadata_tags_lifecycle(self):
    path = self._url('/v2/metadefs/namespaces')
    headers = self._headers({'content-type': 'application/json'})
    tenant1_namespaces = []
    tenant2_namespaces = []
    for tenant in [self.tenant1, self.tenant2]:
        headers['X-Tenant-Id'] = tenant
        for visibility in ['public', 'private']:
            namespace_data = {'namespace': '%s_%s_namespace' % (tenant, visibility), 'display_name': 'My User Friendly Namespace', 'description': 'My description', 'visibility': visibility, 'owner': tenant}
            namespace = self.create_namespace(path, headers, namespace_data)
            self.assertNamespacesEqual(namespace, namespace_data)
            if tenant == self.tenant1:
                tenant1_namespaces.append(namespace)
            else:
                tenant2_namespaces.append(namespace)
    tenant1_tags = self._create_tags(tenant1_namespaces)
    tenant2_tags = self._create_tags(tenant2_namespaces)

    def _check_tag_access(tags, tenant):
        headers = self._headers({'content-type': 'application/json', 'X-Tenant-Id': tenant, 'X-Roles': 'reader,member'})
        for tag in tags:
            for namespace, tag_name in tag.items():
                path = self._url('/v2/metadefs/namespaces/%s/tags/%s' % (namespace, tag_name))
                response = requests.get(path, headers=headers)
                if namespace.split('_')[1] == 'public':
                    expected = http.OK
                else:
                    expected = http.NOT_FOUND
                self.assertEqual(expected, response.status_code)
                path = self._url('/v2/metadefs/namespaces/%s/tags' % namespace)
                response = requests.get(path, headers=headers)
                self.assertEqual(expected, response.status_code)
                if expected == http.OK:
                    resp_props = response.json()['tags']
                    self.assertEqual(sorted(tag.values()), sorted([x['name'] for x in resp_props]))
    _check_tag_access(tenant2_tags, self.tenant1)
    _check_tag_access(tenant1_tags, self.tenant2)
    total_tags = tenant1_tags + tenant2_tags
    for tag in total_tags:
        for namespace, tag_name in tag.items():
            data = {'name': tag_name}
            path = self._url('/v2/metadefs/namespaces/%s/tags/%s' % (namespace, tag_name))
            headers['X-Roles'] = 'reader,member'
            response = requests.put(path, headers=headers, json=data)
            self.assertEqual(http.FORBIDDEN, response.status_code)
            headers = self._headers({'X-Tenant-Id': namespace.split('_')[0]})
            self._update_tags(path, headers, data)
    for tag in total_tags:
        for namespace, tag_name in tag.items():
            path = self._url('/v2/metadefs/namespaces/%s/tags/%s' % (namespace, tag_name))
            response = requests.delete(path, headers=self._headers({'X-Roles': 'reader,member', 'X-Tenant-Id': namespace.split('_')[0]}))
            self.assertEqual(http.FORBIDDEN, response.status_code)
    headers = self._headers()
    for tag in total_tags:
        for namespace, tag_name in tag.items():
            path = self._url('/v2/metadefs/namespaces/%s/tags/%s' % (namespace, tag_name))
            response = requests.delete(path, headers=headers)
            self.assertEqual(http.NO_CONTENT, response.status_code)
            response = requests.get(path, headers=headers)
            self.assertEqual(http.NOT_FOUND, response.status_code)
    headers = self._headers({'content-type': 'application/json', 'X-Roles': 'reader,member'})
    data = {'tags': [{'name': 'tag1'}, {'name': 'tag2'}, {'name': 'tag3'}]}
    for namespace in tenant1_namespaces:
        path = self._url('/v2/metadefs/namespaces/%s/tags' % namespace['namespace'])
        response = requests.post(path, headers=headers, json=data)
        self.assertEqual(http.FORBIDDEN, response.status_code)
    headers = self._headers({'content-type': 'application/json'})
    for namespace in tenant1_namespaces:
        path = self._url('/v2/metadefs/namespaces/%s/tags' % namespace['namespace'])
        response = requests.post(path, headers=headers, json=data)
        self.assertEqual(http.CREATED, response.status_code)
    headers = self._headers({'content-type': 'application/json', 'X-Roles': 'reader,member'})
    for namespace in tenant1_namespaces:
        path = self._url('/v2/metadefs/namespaces/%s/tags' % namespace['namespace'])
        response = requests.delete(path, headers=headers)
        self.assertEqual(http.FORBIDDEN, response.status_code)
    headers = self._headers()
    for namespace in tenant1_namespaces:
        path = self._url('/v2/metadefs/namespaces/%s/tags' % namespace['namespace'])
        response = requests.delete(path, headers=headers)
        self.assertEqual(http.NO_CONTENT, response.status_code)