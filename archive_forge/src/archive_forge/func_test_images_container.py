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
def test_images_container(self):
    self.start_servers(**self.__dict__.copy())
    path = self._url('/v2/images')
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    images = jsonutils.loads(response.text)['images']
    first = jsonutils.loads(response.text)['first']
    self.assertEqual(0, len(images))
    self.assertNotIn('next', jsonutils.loads(response.text))
    self.assertEqual('/v2/images', first)
    images = []
    fixtures = [{'name': 'image-3', 'type': 'kernel', 'ping': 'pong', 'container_format': 'ami', 'disk_format': 'ami'}, {'name': 'image-4', 'type': 'kernel', 'ping': 'pong', 'container_format': 'bare', 'disk_format': 'ami'}, {'name': 'image-1', 'type': 'kernel', 'ping': 'pong'}, {'name': 'image-3', 'type': 'ramdisk', 'ping': 'pong'}, {'name': 'image-2', 'type': 'kernel', 'ping': 'ding'}, {'name': 'image-3', 'type': 'kernel', 'ping': 'pong'}, {'name': 'image-2,image-5', 'type': 'kernel', 'ping': 'pong'}]
    path = self._url('/v2/images')
    headers = self._headers({'content-type': 'application/json'})
    for fixture in fixtures:
        data = jsonutils.dumps(fixture)
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        images.append(jsonutils.loads(response.text))
    path = self._url('/v2/images')
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    body = jsonutils.loads(response.text)
    self.assertEqual(7, len(body['images']))
    self.assertEqual('/v2/images', body['first'])
    self.assertNotIn('next', jsonutils.loads(response.text))
    url_template = '/v2/images?created_at=lt:%s'
    path = self._url(url_template % images[0]['created_at'])
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    body = jsonutils.loads(response.text)
    self.assertEqual(0, len(body['images']))
    self.assertEqual(url_template % images[0]['created_at'], urllib.parse.unquote(body['first']))
    url_template = '/v2/images?updated_at=lt:%s'
    path = self._url(url_template % images[2]['updated_at'])
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    body = jsonutils.loads(response.text)
    self.assertGreaterEqual(3, len(body['images']))
    self.assertEqual(url_template % images[2]['updated_at'], urllib.parse.unquote(body['first']))
    url_template = '/v2/images?%s=lt:invalid_value'
    for filter in ['updated_at', 'created_at']:
        path = self._url(url_template % filter)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.BAD_REQUEST, response.status_code)
    url_template = '/v2/images?%s=invalid_operator:2015-11-19T12:24:02Z'
    for filter in ['updated_at', 'created_at']:
        path = self._url(url_template % filter)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.BAD_REQUEST, response.status_code)
    path = self._url('/v2/images?name=%FF')
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.BAD_REQUEST, response.status_code)
    url_template = '/v2/images?name=in:%s'
    filter_value = 'image-1,image-2'
    path = self._url(url_template % filter_value)
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    body = jsonutils.loads(response.text)
    self.assertGreaterEqual(3, len(body['images']))
    url_template = '/v2/images?container_format=in:%s'
    filter_value = 'bare,ami'
    path = self._url(url_template % filter_value)
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    body = jsonutils.loads(response.text)
    self.assertGreaterEqual(2, len(body['images']))
    url_template = '/v2/images?disk_format=in:%s'
    filter_value = 'bare,ami,iso'
    path = self._url(url_template % filter_value)
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    body = jsonutils.loads(response.text)
    self.assertGreaterEqual(2, len(body['images']))
    template_url = '/v2/images?limit=2&sort_dir=asc&sort_key=name&marker=%s&type=kernel&ping=pong'
    path = self._url(template_url % images[2]['id'])
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    body = jsonutils.loads(response.text)
    self.assertEqual(2, len(body['images']))
    response_ids = [image['id'] for image in body['images']]
    self.assertEqual([images[6]['id'], images[0]['id']], response_ids)
    path = self._url(body['next'])
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    body = jsonutils.loads(response.text)
    self.assertEqual(2, len(body['images']))
    response_ids = [image['id'] for image in body['images']]
    self.assertEqual([images[5]['id'], images[1]['id']], response_ids)
    path = self._url(body['next'])
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    body = jsonutils.loads(response.text)
    self.assertEqual(0, len(body['images']))
    path = self._url('/v2/images/%s' % images[0]['id'])
    response = requests.delete(path, headers=self._headers())
    self.assertEqual(http.NO_CONTENT, response.status_code)
    path = self._url('/v2/images?marker=%s' % images[0]['id'])
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.BAD_REQUEST, response.status_code)
    self.stop_servers()