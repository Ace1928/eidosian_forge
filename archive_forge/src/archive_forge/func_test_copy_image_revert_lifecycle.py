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
def test_copy_image_revert_lifecycle(self):
    self.start_servers(**self.__dict__.copy())
    path = self._url('/v2/images')
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    images = jsonutils.loads(response.text)['images']
    self.assertEqual(0, len(images))
    path = self._url('/v2/info/import')
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    discovery_calls = jsonutils.loads(response.text)['import-methods']['value']
    self.assertIn('copy-image', discovery_calls)
    available_stores = ['file1', 'file2', 'file3']
    path = self._url('/v2/info/stores')
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    discovery_calls = jsonutils.loads(response.text)['stores']
    for stores in discovery_calls:
        self.assertIn('id', stores)
        self.assertIn(stores['id'], available_stores)
        self.assertFalse(stores['id'].startswith('os_glance_'))
    path = self._url('/v2/images')
    headers = self._headers({'content-type': 'application/json'})
    data = jsonutils.dumps({'name': 'image-1', 'type': 'kernel', 'disk_format': 'aki', 'container_format': 'aki'})
    response = requests.post(path, headers=headers, data=data)
    self.assertEqual(http.CREATED, response.status_code)
    self.assertIn('OpenStack-image-store-ids', response.headers)
    for store in available_stores:
        self.assertIn(store, response.headers['OpenStack-image-store-ids'])
    image = jsonutils.loads(response.text)
    image_id = image['id']
    checked_keys = set(['status', 'name', 'tags', 'created_at', 'updated_at', 'visibility', 'self', 'protected', 'id', 'file', 'min_disk', 'type', 'min_ram', 'schema', 'disk_format', 'container_format', 'owner', 'checksum', 'size', 'virtual_size', 'os_hidden', 'os_hash_algo', 'os_hash_value'])
    self.assertEqual(checked_keys, set(image.keys()))
    expected_image = {'status': 'queued', 'name': 'image-1', 'tags': [], 'visibility': 'shared', 'self': '/v2/images/%s' % image_id, 'protected': False, 'file': '/v2/images/%s/file' % image_id, 'min_disk': 0, 'type': 'kernel', 'min_ram': 0, 'schema': '/v2/schemas/image'}
    for key, value in expected_image.items():
        self.assertEqual(value, image[key], key)
    path = self._url('/v2/images')
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    images = jsonutils.loads(response.text)['images']
    self.assertEqual(1, len(images))
    self.assertEqual(image_id, images[0]['id'])
    func_utils.verify_image_hashes_and_status(self, image_id, status='queued')
    path = self._url('/v2/images/%s/import' % image_id)
    headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
    thread, httpd, port = test_utils.start_standalone_http_server()
    image_data_uri = 'http://localhost:%s/' % port
    data = jsonutils.dumps({'method': {'name': 'web-download', 'uri': image_data_uri}, 'stores': ['file1']})
    response = requests.post(path, headers=headers, data=data)
    self.assertEqual(http.ACCEPTED, response.status_code)
    path = self._url('/v2/images/%s' % image_id)
    func_utils.wait_for_status(request_path=path, request_headers=self._headers(), status='active', max_sec=40, delay_sec=0.2, start_delay_sec=1)
    with requests.get(image_data_uri) as r:
        expect_c = str(md5(r.content, usedforsecurity=False).hexdigest())
        expect_h = str(hashlib.sha512(r.content).hexdigest())
    func_utils.verify_image_hashes_and_status(self, image_id, checksum=expect_c, os_hash_value=expect_h, size=len(r.content), status='active')
    httpd.shutdown()
    httpd.server_close()
    path = self._url('/v2/images/%s' % image_id)
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    self.assertIn('file1', jsonutils.loads(response.text)['stores'])
    path = self._url('/v2/images/%s/import' % image_id)
    headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
    os.rmdir(self.test_dir + '/images_3')
    data = jsonutils.dumps({'method': {'name': 'copy-image'}, 'stores': ['file2', 'file3']})
    response = requests.post(path, headers=headers, data=data)
    self.assertEqual(http.ACCEPTED, response.status_code)

    def poll_callback(image):
        return not (image['os_glance_importing_to_stores'] == '' and image['os_glance_failed_import'] == 'file3' and (image['stores'] == 'file1'))
    func_utils.poll_entity(self._url('/v2/images/%s' % image_id), self._headers(), poll_callback)
    path = self._url('/v2/images/%s' % image_id)
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    self.assertIn('file1', jsonutils.loads(response.text)['stores'])
    self.assertNotIn('file2', jsonutils.loads(response.text)['stores'])
    self.assertNotIn('file3', jsonutils.loads(response.text)['stores'])
    fail_key = 'os_glance_failed_import'
    pend_key = 'os_glance_importing_to_stores'
    self.assertEqual('file3', jsonutils.loads(response.text)[fail_key])
    self.assertEqual('', jsonutils.loads(response.text)[pend_key])
    path = self._url('/v2/images/%s/import' % image_id)
    headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
    data = jsonutils.dumps({'method': {'name': 'copy-image'}, 'stores': ['file2', 'file3'], 'all_stores_must_succeed': False})
    for i in range(0, 5):
        response = requests.post(path, headers=headers, data=data)
        if response.status_code != http.CONFLICT:
            break
        time.sleep(1)
    self.assertEqual(http.ACCEPTED, response.status_code)
    path = self._url('/v2/images/%s' % image_id)
    func_utils.wait_for_copying(request_path=path, request_headers=self._headers(), stores=['file2'], max_sec=10, delay_sec=0.2, start_delay_sec=1, failure_scenario=True)
    path = self._url('/v2/images/%s' % image_id)
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    self.assertIn('file1', jsonutils.loads(response.text)['stores'])
    self.assertIn('file2', jsonutils.loads(response.text)['stores'])
    self.assertNotIn('file3', jsonutils.loads(response.text)['stores'])
    path = self._url('/v2/images/%s' % image_id)
    response = requests.delete(path, headers=self._headers())
    self.assertEqual(http.NO_CONTENT, response.status_code)
    path = self._url('/v2/images')
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    images = jsonutils.loads(response.text)['images']
    self.assertEqual(0, len(images))
    self.stop_servers()