import http.client as http
from oslo_serialization import jsonutils
import requests
from glance.tests.functional.v2 import metadef_base
def test_metadata_objects_lifecycle(self):
    path = self._url('/v2/metadefs/namespaces/MyNamespace')
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.NOT_FOUND, response.status_code)
    path = self._url('/v2/metadefs/namespaces')
    headers = self._headers({'content-type': 'application/json'})
    namespace_name = 'MyNamespace'
    data = jsonutils.dumps({'namespace': namespace_name, 'display_name': 'My User Friendly Namespace', 'description': 'My description', 'visibility': 'public', 'protected': False, 'owner': 'The Test Owner'})
    response = requests.post(path, headers=headers, data=data)
    self.assertEqual(http.CREATED, response.status_code)
    path = self._url('/v2/metadefs/namespaces/MyNamespace/objects/object1')
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.NOT_FOUND, response.status_code)
    path = self._url('/v2/metadefs/namespaces/MyNamespace/objects')
    headers = self._headers({'content-type': 'application/json'})
    metadata_object_name = 'object1'
    data = jsonutils.dumps({'name': metadata_object_name, 'description': 'object1 description.', 'required': ['property1'], 'properties': {'property1': {'type': 'integer', 'title': 'property1', 'description': 'property1 description', 'operators': ['<all-in>'], 'default': 100, 'minimum': 100, 'maximum': 30000369}, 'property2': {'type': 'string', 'title': 'property2', 'description': 'property2 description ', 'default': 'value2', 'minLength': 2, 'maxLength': 50}}})
    response = requests.post(path, headers=headers, data=data)
    self.assertEqual(http.CREATED, response.status_code)
    response = requests.post(path, headers=headers, data=data)
    self.assertEqual(http.CONFLICT, response.status_code)
    path = self._url('/v2/metadefs/namespaces/%s/objects/%s' % (namespace_name, metadata_object_name))
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    metadata_object = jsonutils.loads(response.text)
    self.assertEqual('object1', metadata_object['name'])
    metadata_object = jsonutils.loads(response.text)
    checked_keys = set(['name', 'description', 'properties', 'required', 'self', 'schema', 'created_at', 'updated_at'])
    self.assertEqual(set(metadata_object.keys()), checked_keys)
    expected_metadata_object = {'name': metadata_object_name, 'description': 'object1 description.', 'required': ['property1'], 'properties': {'property1': {'type': 'integer', 'title': 'property1', 'description': 'property1 description', 'operators': ['<all-in>'], 'default': 100, 'minimum': 100, 'maximum': 30000369}, 'property2': {'type': 'string', 'title': 'property2', 'description': 'property2 description ', 'default': 'value2', 'minLength': 2, 'maxLength': 50}}, 'self': '/v2/metadefs/namespaces/%(namespace)s/objects/%(object)s' % {'namespace': namespace_name, 'object': metadata_object_name}, 'schema': 'v2/schemas/metadefs/object'}
    checked_values = set(['name', 'description'])
    for key, value in expected_metadata_object.items():
        if key in checked_values:
            self.assertEqual(metadata_object[key], value, key)
    for key, value in expected_metadata_object['properties']['property2'].items():
        self.assertEqual(metadata_object['properties']['property2'][key], value, key)
    path = self._url('/v2/metadefs/namespaces/%s/objects/%s' % (namespace_name, metadata_object_name))
    media_type = 'application/json'
    headers = self._headers({'content-type': media_type})
    metadata_object_name = 'object1-UPDATED'
    data = jsonutils.dumps({'name': metadata_object_name, 'description': 'desc-UPDATED', 'required': ['property2'], 'properties': {'property1': {'type': 'integer', 'title': 'property1', 'description': 'p1 desc-UPDATED', 'default': 500, 'minimum': 500, 'maximum': 1369}, 'property2': {'type': 'string', 'title': 'property2', 'description': 'p2 desc-UPDATED', 'operators': ['<or>'], 'default': 'value2-UPDATED', 'minLength': 5, 'maxLength': 150}}})
    response = requests.put(path, headers=headers, data=data)
    self.assertEqual(http.OK, response.status_code, response.text)
    metadata_object = jsonutils.loads(response.text)
    self.assertEqual('object1-UPDATED', metadata_object['name'])
    self.assertEqual('desc-UPDATED', metadata_object['description'])
    self.assertEqual('property2', metadata_object['required'][0])
    updated_property1 = metadata_object['properties']['property1']
    updated_property2 = metadata_object['properties']['property2']
    self.assertEqual('integer', updated_property1['type'])
    self.assertEqual('p1 desc-UPDATED', updated_property1['description'])
    self.assertEqual('500', updated_property1['default'])
    self.assertEqual(500, updated_property1['minimum'])
    self.assertEqual(1369, updated_property1['maximum'])
    self.assertEqual(['<or>'], updated_property2['operators'])
    self.assertEqual('string', updated_property2['type'])
    self.assertEqual('p2 desc-UPDATED', updated_property2['description'])
    self.assertEqual('value2-UPDATED', updated_property2['default'])
    self.assertEqual(5, updated_property2['minLength'])
    self.assertEqual(150, updated_property2['maxLength'])
    path = self._url('/v2/metadefs/namespaces/%s/objects/%s' % (namespace_name, metadata_object_name))
    response = requests.get(path, headers=self._headers())
    self.assertEqual(200, response.status_code)
    self.assertEqual('object1-UPDATED', metadata_object['name'])
    self.assertEqual('desc-UPDATED', metadata_object['description'])
    self.assertEqual('property2', metadata_object['required'][0])
    updated_property1 = metadata_object['properties']['property1']
    updated_property2 = metadata_object['properties']['property2']
    self.assertEqual('integer', updated_property1['type'])
    self.assertEqual('p1 desc-UPDATED', updated_property1['description'])
    self.assertEqual('500', updated_property1['default'])
    self.assertEqual(500, updated_property1['minimum'])
    self.assertEqual(1369, updated_property1['maximum'])
    self.assertEqual(['<or>'], updated_property2['operators'])
    self.assertEqual('string', updated_property2['type'])
    self.assertEqual('p2 desc-UPDATED', updated_property2['description'])
    self.assertEqual('value2-UPDATED', updated_property2['default'])
    self.assertEqual(5, updated_property2['minLength'])
    self.assertEqual(150, updated_property2['maxLength'])
    path = self._url('/v2/metadefs/namespaces/%s/objects/%s' % (namespace_name, metadata_object_name))
    response = requests.delete(path, headers=self._headers())
    self.assertEqual(http.NO_CONTENT, response.status_code)
    path = self._url('/v2/metadefs/namespaces/%s/objects/%s' % (namespace_name, metadata_object_name))
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.NOT_FOUND, response.status_code)