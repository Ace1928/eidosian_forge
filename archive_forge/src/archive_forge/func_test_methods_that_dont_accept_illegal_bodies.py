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
def test_methods_that_dont_accept_illegal_bodies(self):
    self.start_servers(**self.__dict__.copy())
    path = self._url('/v2/images')
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    schema_urls = ['/v2/schemas/images', '/v2/schemas/image', '/v2/schemas/members', '/v2/schemas/member']
    for value in schema_urls:
        path = self._url(value)
        data = jsonutils.dumps(['body'])
        response = requests.get(path, headers=self._headers(), data=data)
        self.assertEqual(http.BAD_REQUEST, response.status_code)
    path = self._url('/v2/images')
    headers = self._headers({'content-type': 'application/json'})
    data = jsonutils.dumps({'name': 'image'})
    response = requests.post(path, headers=headers, data=data)
    self.assertEqual(http.CREATED, response.status_code)
    image = jsonutils.loads(response.text)
    image_id = image['id']
    test_urls = [('/v2/images/%s', 'get'), ('/v2/images/%s/actions/deactivate', 'post'), ('/v2/images/%s/actions/reactivate', 'post'), ('/v2/images/%s/tags/mytag', 'put'), ('/v2/images/%s/tags/mytag', 'delete'), ('/v2/images/%s/members', 'get'), ('/v2/images/%s/file', 'get'), ('/v2/images/%s', 'delete')]
    for link, method in test_urls:
        path = self._url(link % image_id)
        data = jsonutils.dumps(['body'])
        response = getattr(requests, method)(path, headers=self._headers(), data=data)
        self.assertEqual(http.BAD_REQUEST, response.status_code)
    path = self._url('/v2/images/%s' % image_id)
    data = '{"hello"]'
    response = requests.delete(path, headers=self._headers(), data=data)
    self.assertEqual(http.BAD_REQUEST, response.status_code)
    path = self._url('/v2/images/%s/members' % image_id)
    data = jsonutils.dumps({'member': TENANT3})
    response = requests.post(path, headers=self._headers(), data=data)
    self.assertEqual(http.OK, response.status_code)
    path = self._url('/v2/images/%s/members/%s' % (image_id, TENANT3))
    data = jsonutils.dumps(['body'])
    response = requests.get(path, headers=self._headers(), data=data)
    self.assertEqual(http.BAD_REQUEST, response.status_code)
    path = self._url('/v2/images/%s/members/%s' % (image_id, TENANT3))
    data = jsonutils.dumps(['body'])
    response = requests.delete(path, headers=self._headers(), data=data)
    self.assertEqual(http.BAD_REQUEST, response.status_code)
    self.stop_servers()