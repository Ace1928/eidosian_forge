import http.client as http
from oslo_serialization import jsonutils
import requests
from glance.tests.functional.v2 import metadef_base
def test_role_base_metadef_resource_types_lifecycle(self):
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
    tenant1_resource_types = self._create_resource_type(tenant1_namespaces)
    tenant2_resource_types = self._create_resource_type(tenant2_namespaces)

    def _check_resource_type_access(namespaces, tenant):
        headers = self._headers({'X-Tenant-Id': tenant, 'X-Roles': 'reader,member'})
        for namespace in namespaces:
            path = self._url('/v2/metadefs/namespaces/%s/resource_types' % namespace['namespace'])
            response = requests.get(path, headers=headers)
            if namespace['visibility'] == 'public':
                self.assertEqual(http.OK, response.status_code)
            else:
                self.assertEqual(http.NOT_FOUND, response.status_code)

    def _check_resource_types(tenant, total_rs_types):
        path = self._url('/v2/metadefs/resource_types')
        headers = self._headers({'X-Tenant-Id': tenant, 'X-Roles': 'reader,member'})
        response = requests.get(path, headers=headers)
        self.assertEqual(http.OK, response.status_code)
        metadef_resource_type = response.json()
        self.assertEqual(sorted((x['name'] for x in metadef_resource_type['resource_types'])), sorted((value for x in total_rs_types for key, value in x.items())))
    _check_resource_type_access(tenant2_namespaces, self.tenant1)
    _check_resource_type_access(tenant1_namespaces, self.tenant2)
    total_resource_types = tenant1_resource_types + tenant2_resource_types
    _check_resource_types(self.tenant1, total_resource_types)
    _check_resource_types(self.tenant2, total_resource_types)
    for resource_type in total_resource_types:
        for namespace, rs_type in resource_type.items():
            path = self._url('/v2/metadefs/namespaces/%s/resource_types/%s' % (namespace, rs_type))
            response = requests.delete(path, headers=self._headers({'X-Roles': 'reader,member', 'X-Tenant-Id': namespace.split('_')[0]}))
            self.assertEqual(http.FORBIDDEN, response.status_code)
    headers = self._headers()
    for resource_type in total_resource_types:
        for namespace, rs_type in resource_type.items():
            path = self._url('/v2/metadefs/namespaces/%s/resource_types/%s' % (namespace, rs_type))
            response = requests.delete(path, headers=headers)
            self.assertEqual(http.NO_CONTENT, response.status_code)
            path = self._url('/v2/metadefs/namespaces/%s/resource_types' % namespace)
            response = requests.get(path, headers=headers)
            metadef_resource_type = response.json()
            self.assertEqual([], metadef_resource_type['resource_type_associations'])