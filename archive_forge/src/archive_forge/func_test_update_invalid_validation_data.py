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
def test_update_invalid_validation_data(self):
    request = self._get_fake_patch_request()
    changes = [{'op': 'add', 'path': '/locations/0', 'value': {'url': 'http://localhost/fake', 'metadata': {}}}]
    changes[0]['value']['validation_data'] = {'os_hash_algo': 'sha512', 'os_hash_value': MULTIHASH1, 'checksum': CHKSUM}
    request.body = jsonutils.dump_as_bytes(changes)
    self.deserializer.update(request)
    changes[0]['value']['validation_data'] = {'os_hash_algo': 'sha512', 'os_hash_value': MULTIHASH1, 'checksum': CHKSUM, 'bogus_key': 'bogus_value'}
    request.body = jsonutils.dump_as_bytes(changes)
    self.assertRaisesRegex(webob.exc.HTTPBadRequest, 'Additional properties are not allowed', self.deserializer.update, request)
    changes[0]['value']['validation_data'] = {'checksum': CHKSUM}
    request.body = jsonutils.dump_as_bytes(changes)
    self.assertRaisesRegex(webob.exc.HTTPBadRequest, 'os_hash.* is a required property', self.deserializer.update, request)