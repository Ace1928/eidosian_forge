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
def test_image_size_cap(self):
    self.api_server.image_size_cap = 128
    self.start_servers(**self.__dict__.copy())
    path = self._url('/v2/images')
    headers = self._headers({'content-type': 'application/json'})
    data = jsonutils.dumps({'name': 'image-size-cap-test-image', 'type': 'kernel', 'disk_format': 'aki', 'container_format': 'aki'})
    response = requests.post(path, headers=headers, data=data)
    self.assertEqual(http.CREATED, response.status_code)
    image = jsonutils.loads(response.text)
    image_id = image['id']
    path = self._url('/v2/images/%s/file' % image_id)
    headers = self._headers({'Content-Type': 'application/octet-stream'})

    class StreamSim(object):

        def __init__(self, size):
            self.size = size

        def __iter__(self):
            yield (b'Z' * self.size)
    response = requests.put(path, headers=headers, data=StreamSim(self.api_server.image_size_cap + 1))
    self.assertEqual(http.REQUEST_ENTITY_TOO_LARGE, response.status_code)
    path = self._url('/v2/images/{0}'.format(image_id))
    headers = self._headers({'content-type': 'application/json'})
    response = requests.get(path, headers=headers)
    image_checksum = jsonutils.loads(response.text).get('checksum')
    self.assertNotEqual(image_checksum, '76522d28cb4418f12704dfa7acd6e7ee')