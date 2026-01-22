import http.client as http
from oslo_serialization import jsonutils
import requests
from glance.tests.functional.v2 import metadef_base
def test_role_based_namespace_lifecycle(self):
    path = self._url('/v2/metadefs/namespaces')
    headers = self._headers({'content-type': 'application/json'})
    tenant_namespaces = dict()
    for tenant in [self.tenant1, self.tenant2]:
        headers['X-Tenant-Id'] = tenant
        for visibility in ['public', 'private']:
            namespace_data = {'namespace': '%s_%s_namespace' % (tenant, visibility), 'display_name': 'My User Friendly Namespace', 'description': 'My description', 'visibility': visibility, 'owner': tenant}
            namespace = self.create_namespace(path, headers, namespace_data)
            self.assertNamespacesEqual(namespace, namespace_data)
            tenant_namespaces.setdefault(tenant, list())
            tenant_namespaces[tenant].append(namespace)

    def _get_expected_namespaces(tenant):
        expected_namespaces = []
        for x in tenant_namespaces[tenant]:
            expected_namespaces.append(x['namespace'])
        if tenant == self.tenant1:
            expected_namespaces.append(tenant_namespaces[self.tenant2][0]['namespace'])
        else:
            expected_namespaces.append(tenant_namespaces[self.tenant1][0]['namespace'])
        return expected_namespaces
    for tenant in [self.tenant1, self.tenant2]:
        path = self._url('/v2/metadefs/namespaces')
        headers = self._headers({'X-Tenant-Id': tenant, 'X-Roles': 'reader,member'})
        response = requests.get(path, headers=headers)
        self.assertEqual(http.OK, response.status_code)
        namespaces = response.json()['namespaces']
        expected_namespaces = _get_expected_namespaces(tenant)
        self.assertEqual(sorted((x['namespace'] for x in namespaces)), sorted(expected_namespaces))

    def _check_namespace_access(namespaces, tenant):
        headers = self._headers({'X-Tenant-Id': tenant, 'X-Roles': 'reader,member'})
        for namespace in namespaces:
            path = self._url('/v2/metadefs/namespaces/%s' % namespace['namespace'])
            headers = headers
            response = requests.get(path, headers=headers)
            if namespace['visibility'] == 'public':
                self.assertEqual(http.OK, response.status_code)
            else:
                self.assertEqual(http.NOT_FOUND, response.status_code)
    _check_namespace_access(tenant_namespaces[self.tenant2], self.tenant1)
    _check_namespace_access(tenant_namespaces[self.tenant1], self.tenant2)
    total_ns = tenant_namespaces[self.tenant1] + tenant_namespaces[self.tenant2]
    for namespace in total_ns:
        data = {'namespace': namespace['namespace'], 'display_name': 'display_name-UPDATED', 'description': 'description-UPDATED', 'visibility': namespace['visibility'], 'protected': True, 'owner': namespace['owner']}
        path = self._url('/v2/metadefs/namespaces/%s' % namespace['namespace'])
        headers = self._headers({'X-Tenant-Id': namespace['owner']})
        headers['X-Roles'] = 'reader,member'
        response = requests.put(path, headers=headers, json=data)
        self.assertEqual(http.FORBIDDEN, response.status_code)
        headers['X-Roles'] = 'admin'
        namespace = self._update_namespace(path, headers, data)
        path = self._url('/v2/metadefs/namespaces/%s' % namespace['namespace'])
        headers['X-Roles'] = 'admin'
        response = requests.delete(path, headers=headers)
        self.assertEqual(http.FORBIDDEN, response.status_code)
        path = self._url('/v2/metadefs/namespaces/%s' % namespace['namespace'])
        response = requests.delete(path, headers=self._headers({'X-Roles': 'reader,member', 'X-Tenant-Id': namespace['owner']}))
        self.assertEqual(http.FORBIDDEN, response.status_code)
    headers = self._headers()
    for namespace in total_ns:
        path = self._url('/v2/metadefs/namespaces/%s' % namespace['namespace'])
        headers = headers
        data = {'namespace': namespace['namespace'], 'protected': False}
        response = requests.put(path, headers=headers, json=data)
        self.assertEqual(http.OK, response.status_code)
    path = self._url('/v2/metadefs/namespaces')
    response = requests.get(path, headers=headers)
    self.assertEqual(http.OK, response.status_code)
    self.assertFalse(namespace['protected'])
    namespaces = response.json()['namespaces']
    for namespace in namespaces:
        path = self._url('/v2/metadefs/namespaces/%s' % namespace['namespace'])
        response = requests.delete(path, headers=self._headers({'X-Roles': 'reader,member', 'X-Tenant-Id': namespace['owner']}))
        self.assertEqual(http.FORBIDDEN, response.status_code)
    for namespace in total_ns:
        path = self._url('/v2/metadefs/namespaces/%s' % namespace['namespace'])
        response = requests.delete(path, headers=headers)
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/metadefs/namespaces/%s' % namespace['namespace'])
        response = requests.get(path, headers=headers)
        self.assertEqual(http.NOT_FOUND, response.status_code)