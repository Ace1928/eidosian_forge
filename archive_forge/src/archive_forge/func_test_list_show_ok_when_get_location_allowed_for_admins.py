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
def test_list_show_ok_when_get_location_allowed_for_admins(self):
    self.api_server.show_image_direct_url = True
    self.api_server.show_multiple_locations = True
    rules = {'context_is_admin': 'role:admin', 'default': '', 'add_image': '', 'get_image': '', 'modify_image': '', 'upload_image': '', 'get_image_location': 'role:admin', 'delete_image': '', 'restricted': '', 'download_image': '', 'add_member': ''}
    self.set_policy_rules(rules)
    self.start_servers(**self.__dict__.copy())
    path = self._url('/v2/images')
    headers = self._headers({'content-type': 'application/json', 'X-Tenant-Id': TENANT1})
    data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki'})
    response = requests.post(path, headers=headers, data=data)
    self.assertEqual(http.CREATED, response.status_code)
    image = jsonutils.loads(response.text)
    image_id = image['id']
    path = self._url('/v2/images/%s' % image_id)
    response = requests.get(path, headers=headers)
    self.assertEqual(http.OK, response.status_code)
    path = self._url('/v2/images')
    response = requests.get(path, headers=headers)
    self.assertEqual(http.OK, response.status_code)
    self.stop_servers()