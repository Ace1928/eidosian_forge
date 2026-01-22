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
def test_download_random_access_w_content_range(self):
    """
        Even though Content-Range is incorrect on requests, we support it
        for backward compatibility with clients written for pre-Pike Glance.
        The following test is for 'Content-Range' requests, which we have
        to ensure that we prevent regression.
        """
    self.start_servers(**self.__dict__.copy())
    path = self._url('/v2/images')
    headers = self._headers({'content-type': 'application/json'})
    data = jsonutils.dumps({'name': 'image-2', 'type': 'kernel', 'bar': 'foo', 'disk_format': 'aki', 'container_format': 'aki', 'xyz': 'abc'})
    response = requests.post(path, headers=headers, data=data)
    self.assertEqual(http.CREATED, response.status_code)
    image = jsonutils.loads(response.text)
    image_id = image['id']
    image_data = 'Z' * 15
    path = self._url('/v2/images/%s/file' % image_id)
    headers = self._headers({'Content-Type': 'application/octet-stream'})
    response = requests.put(path, headers=headers, data=image_data)
    self.assertEqual(http.NO_CONTENT, response.status_code)
    result_body = ''
    for x in range(15):
        content_range = 'bytes %s-%s/15' % (x, x)
        headers = self._headers({'Content-Range': content_range})
        path = self._url('/v2/images/%s/file' % image_id)
        response = requests.get(path, headers=headers)
        self.assertEqual(http.PARTIAL_CONTENT, response.status_code)
        result_body += response.text
    self.assertEqual(result_body, image_data)
    content_range = 'bytes 3-16/15'
    headers = self._headers({'Content-Range': content_range})
    path = self._url('/v2/images/%s/file' % image_id)
    response = requests.get(path, headers=headers)
    self.assertEqual(http.REQUESTED_RANGE_NOT_SATISFIABLE, response.status_code)
    self.stop_servers()