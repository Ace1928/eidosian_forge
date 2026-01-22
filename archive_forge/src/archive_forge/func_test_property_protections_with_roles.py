import hashlib
import http.client as http
import os
import subprocess
import tempfile
import time
import urllib
import uuid
import fixtures
from oslo_limit import exception as ol_exc
from oslo_limit import limit
from oslo_serialization import jsonutils
from oslo_utils.secretutils import md5
from oslo_utils import units
import requests
from glance.quota import keystone as ks_quota
from glance.tests import functional
from glance.tests.functional import ft_utils as func_utils
from glance.tests import utils as test_utils
def test_property_protections_with_roles(self):
    self.api_server.property_protection_file = self.property_file_roles
    self.start_servers(**self.__dict__.copy())
    path = self._url('/v2/images')
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    images = jsonutils.loads(response.text)['images']
    self.assertEqual(0, len(images))
    path = self._url('/v2/images')
    headers = self._headers({'content-type': 'application/json', 'X-Roles': 'reader,member'})
    data = jsonutils.dumps({'name': 'image-1', 'foo': 'bar', 'disk_format': 'aki', 'container_format': 'aki', 'x_owner_foo': 'o_s_bar'})
    response = requests.post(path, headers=headers, data=data)
    self.assertEqual(http.FORBIDDEN, response.status_code)
    path = self._url('/v2/images')
    headers = self._headers({'content-type': 'application/json', 'X-Roles': 'reader,member'})
    data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki', 'x_owner_foo': 'o_s_bar'})
    response = requests.post(path, headers=headers, data=data)
    self.assertEqual(http.CREATED, response.status_code)
    image = jsonutils.loads(response.text)
    image_id = image['id']
    expected_image = {'status': 'queued', 'name': 'image-1', 'tags': [], 'visibility': 'shared', 'self': '/v2/images/%s' % image_id, 'protected': False, 'file': '/v2/images/%s/file' % image_id, 'min_disk': 0, 'x_owner_foo': 'o_s_bar', 'min_ram': 0, 'schema': '/v2/schemas/image'}
    for key, value in expected_image.items():
        self.assertEqual(value, image[key], key)
    path = self._url('/v2/images')
    headers = self._headers({'content-type': 'application/json', 'X-Roles': 'reader,member,spl_role'})
    data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki', 'spl_create_prop': 'create_bar', 'spl_create_prop_policy': 'create_policy_bar', 'spl_read_prop': 'read_bar', 'spl_update_prop': 'update_bar', 'spl_delete_prop': 'delete_bar', 'spl_delete_empty_prop': ''})
    response = requests.post(path, headers=headers, data=data)
    self.assertEqual(http.CREATED, response.status_code)
    image = jsonutils.loads(response.text)
    image_id = image['id']
    path = self._url('/v2/images/%s' % image_id)
    media_type = 'application/openstack-images-v2.1-json-patch'
    headers = self._headers({'content-type': media_type, 'X-Roles': 'reader,member,spl_role'})
    data = jsonutils.dumps([{'op': 'replace', 'path': '/spl_read_prop', 'value': 'r'}, {'op': 'replace', 'path': '/spl_update_prop', 'value': 'u'}])
    response = requests.patch(path, headers=headers, data=data)
    self.assertEqual(http.FORBIDDEN, response.status_code, response.text)
    path = self._url('/v2/images/%s' % image_id)
    media_type = 'application/openstack-images-v2.1-json-patch'
    headers = self._headers({'content-type': media_type, 'X-Roles': 'reader,member,spl_role'})
    data = jsonutils.dumps([{'op': 'add', 'path': '/spl_new_prop', 'value': 'new'}, {'op': 'remove', 'path': '/spl_create_prop'}, {'op': 'remove', 'path': '/spl_delete_prop'}])
    response = requests.patch(path, headers=headers, data=data)
    self.assertEqual(http.FORBIDDEN, response.status_code, response.text)
    path = self._url('/v2/images/%s' % image_id)
    media_type = 'application/openstack-images-v2.1-json-patch'
    headers = self._headers({'content-type': media_type, 'X-Roles': 'reader,member,spl_role'})
    data = jsonutils.dumps([{'op': 'replace', 'path': '/spl_update_prop', 'value': ''}, {'op': 'replace', 'path': '/spl_update_prop', 'value': 'u'}])
    response = requests.patch(path, headers=headers, data=data)
    self.assertEqual(http.OK, response.status_code, response.text)
    image = jsonutils.loads(response.text)
    self.assertEqual('u', image['spl_update_prop'])
    path = self._url('/v2/images/%s' % image_id)
    media_type = 'application/openstack-images-v2.1-json-patch'
    headers = self._headers({'content-type': media_type, 'X-Roles': 'reader,member,spl_role'})
    data = jsonutils.dumps([{'op': 'remove', 'path': '/spl_delete_prop'}, {'op': 'remove', 'path': '/spl_delete_empty_prop'}])
    response = requests.patch(path, headers=headers, data=data)
    self.assertEqual(http.OK, response.status_code, response.text)
    image = jsonutils.loads(response.text)
    self.assertNotIn('spl_delete_prop', image.keys())
    self.assertNotIn('spl_delete_empty_prop', image.keys())
    path = self._url('/v2/images/%s' % image_id)
    response = requests.delete(path, headers=self._headers())
    self.assertEqual(http.NO_CONTENT, response.status_code)
    path = self._url('/v2/images/%s' % image_id)
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.NOT_FOUND, response.status_code)
    self.stop_servers()