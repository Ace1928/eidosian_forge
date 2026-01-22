import http.client as http
from oslo_serialization import jsonutils
import requests
from glance.tests.functional.v2 import metadef_base
def test_role_base_metadata_objects_lifecycle(self):
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
    tenant1_objects = self._create_object(tenant1_namespaces)
    tenant2_objects = self._create_object(tenant2_namespaces)

    def _check_object_access(objects, tenant):
        headers = self._headers({'content-type': 'application/json', 'X-Tenant-Id': tenant, 'X-Roles': 'reader,member'})
        for obj in objects:
            for namespace, object_name in obj.items():
                path = self._url('/v2/metadefs/namespaces/%s/objects/%s' % (namespace, object_name))
                headers = headers
                response = requests.get(path, headers=headers)
                if namespace.split('_')[1] == 'public':
                    expected = http.OK
                else:
                    expected = http.NOT_FOUND
                self.assertEqual(expected, response.status_code)
                path = self._url('/v2/metadefs/namespaces/%s/objects' % namespace)
                response = requests.get(path, headers=headers)
                self.assertEqual(expected, response.status_code)
                if expected == http.OK:
                    resp_objs = response.json()['objects']
                    self.assertEqual(sorted(obj.values()), sorted([x['name'] for x in resp_objs]))
    _check_object_access(tenant2_objects, self.tenant1)
    _check_object_access(tenant1_objects, self.tenant2)
    total_objects = tenant1_objects + tenant2_objects
    for obj in total_objects:
        for namespace, object_name in obj.items():
            data = {'name': object_name, 'description': 'desc-UPDATED', 'required': ['property1'], 'properties': {'property1': {'type': 'integer', 'title': 'property1', 'description': 'p1 desc-UPDATED'}}}
            path = self._url('/v2/metadefs/namespaces/%s/objects/%s' % (namespace, object_name))
            headers['X-Roles'] = 'reader,member'
            response = requests.put(path, headers=headers, json=data)
            self.assertEqual(http.FORBIDDEN, response.status_code)
            headers = self._headers({'X-Tenant-Id': namespace.split('_')[0]})
            self._update_object(path, headers, data, namespace)
    for obj in total_objects:
        for namespace, object_name in obj.items():
            path = self._url('/v2/metadefs/namespaces/%s/objects/%s' % (namespace, object_name))
            response = requests.delete(path, headers=self._headers({'X-Roles': 'reader,member', 'X-Tenant-Id': namespace.split('_')[0]}))
            self.assertEqual(http.FORBIDDEN, response.status_code)
    headers = self._headers()
    for obj in total_objects:
        for namespace, object_name in obj.items():
            path = self._url('/v2/metadefs/namespaces/%s/objects/%s' % (namespace, object_name))
            response = requests.delete(path, headers=headers)
            self.assertEqual(http.NO_CONTENT, response.status_code)
            response = requests.get(path, headers=headers)
            self.assertEqual(http.NOT_FOUND, response.status_code)