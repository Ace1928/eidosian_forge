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
def test_update_readonly_prop(self):
    self.start_servers(**self.__dict__.copy())
    path = self._url('/v2/images')
    headers = self._headers({'content-type': 'application/json'})
    data = jsonutils.dumps({'name': 'image-1'})
    response = requests.post(path, headers=headers, data=data)
    image = jsonutils.loads(response.text)
    image_id = image['id']
    path = self._url('/v2/images/%s' % image_id)
    media_type = 'application/openstack-images-v2.1-json-patch'
    headers = self._headers({'content-type': media_type})
    props = ['/id', '/file', '/location', '/schema', '/self']
    for prop in props:
        doc = [{'op': 'replace', 'path': prop, 'value': 'value1'}]
        data = jsonutils.dumps(doc)
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.FORBIDDEN, response.status_code)
    for prop in props:
        doc = [{'op': 'remove', 'path': prop, 'value': 'value1'}]
        data = jsonutils.dumps(doc)
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.FORBIDDEN, response.status_code)
    for prop in props:
        doc = [{'op': 'add', 'path': prop, 'value': 'value1'}]
        data = jsonutils.dumps(doc)
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.FORBIDDEN, response.status_code)
    self.stop_servers()