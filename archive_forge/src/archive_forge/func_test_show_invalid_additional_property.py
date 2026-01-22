import datetime
import hashlib
import http.client as http
import os
import requests
from unittest import mock
import uuid
from castellan.common import exception as castellan_exception
import glance_store as store
from oslo_config import cfg
from oslo_serialization import jsonutils
from oslo_utils import fixture
import testtools
import webob
import webob.exc
import glance.api.v2.image_actions
import glance.api.v2.images
from glance.common import exception
from glance.common import store_utils
from glance.common import timeutils
from glance import domain
import glance.notifier
import glance.schema
from glance.tests.unit import base
from glance.tests.unit.keymgr import fake as fake_keymgr
import glance.tests.unit.utils as unit_test_utils
from glance.tests.unit.v2 import test_tasks_resource
import glance.tests.utils as test_utils
def test_show_invalid_additional_property(self):
    """Ensure that the serializer passes
        through invalid additional properties.

        It must not complains with i.e. non-string.
        """
    serializer = glance.api.v2.images.ResponseSerializer()
    self.fixture.extra_properties['marx'] = 123
    expected = {'id': UUID2, 'name': 'image-2', 'status': 'queued', 'visibility': 'private', 'protected': False, 'os_hidden': False, 'checksum': 'ca425b88f047ce8ec45ee90e813ada91', 'os_hash_algo': FAKEHASHALGO, 'os_hash_value': MULTIHASH1, 'marx': 123, 'tags': [], 'size': 1024, 'virtual_size': 3072, 'created_at': ISOTIME, 'updated_at': ISOTIME, 'self': '/v2/images/%s' % UUID2, 'file': '/v2/images/%s/file' % UUID2, 'schema': '/v2/schemas/image', 'owner': '2c014f32-55eb-467d-8fcb-4bd706012f81', 'min_ram': None, 'min_disk': None, 'disk_format': None, 'container_format': None}
    response = webob.Response()
    serializer.show(response, self.fixture)
    self.assertEqual(expected, jsonutils.loads(response.body))