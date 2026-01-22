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
def test_update_locations_with_restricted_sources(self):
    self.api_server.show_multiple_locations = True
    self.start_servers(**self.__dict__.copy())
    path = self._url('/v2/images')
    headers = self._headers({'content-type': 'application/json'})
    data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki'})
    response = requests.post(path, headers=headers, data=data)
    self.assertEqual(http.CREATED, response.status_code)
    image = jsonutils.loads(response.text)
    image_id = image['id']
    self.assertEqual('queued', image['status'])
    self.assertIsNone(image['size'])
    self.assertIsNone(image['virtual_size'])
    path = self._url('/v2/images/%s' % image_id)
    media_type = 'application/openstack-images-v2.1-json-patch'
    headers = self._headers({'content-type': media_type})
    data = jsonutils.dumps([{'op': 'replace', 'path': '/locations', 'value': [{'url': 'file:///foo_image', 'metadata': {}}]}])
    response = requests.patch(path, headers=headers, data=data)
    self.assertEqual(http.BAD_REQUEST, response.status_code, response.text)
    data = jsonutils.dumps([{'op': 'replace', 'path': '/locations', 'value': [{'url': 'swift+config:///foo_image', 'metadata': {}}]}])
    response = requests.patch(path, headers=headers, data=data)
    self.assertEqual(http.BAD_REQUEST, response.status_code, response.text)