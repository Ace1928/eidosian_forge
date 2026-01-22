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
def test_show_full_fixture(self):
    expected = {'id': UUID1, 'name': 'OpenStack™-1', 'status': 'queued', 'visibility': 'public', 'protected': False, 'os_hidden': False, 'tags': set(['Ⅰ', 'Ⅱ']), 'size': 1024, 'virtual_size': 3072, 'checksum': 'ca425b88f047ce8ec45ee90e813ada91', 'os_hash_algo': str(FAKEHASHALGO), 'os_hash_value': str(MULTIHASH1), 'container_format': 'ami', 'disk_format': 'ami', 'min_ram': 128, 'min_disk': 10, 'created_at': str(ISOTIME), 'updated_at': str(ISOTIME), 'self': '/v2/images/%s' % UUID1, 'file': '/v2/images/%s/file' % UUID1, 'schema': '/v2/schemas/image', 'lang': 'Français', 'disposé': 'fâché', 'owner': '6838eb7b-6ded-434a-882c-b344c77fe8df'}
    response = webob.Response()
    self.serializer.show(response, self.fixtures[0])
    actual = jsonutils.loads(response.body)
    actual['tags'] = set(actual['tags'])
    self.assertEqual(expected, actual)
    self.assertEqual('application/json', response.content_type)