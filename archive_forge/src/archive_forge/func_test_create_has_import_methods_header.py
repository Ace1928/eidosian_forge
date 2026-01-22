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
def test_create_has_import_methods_header(self):
    header_name = 'OpenStack-image-import-methods'
    enabled_methods = ['one', 'two', 'three']
    self.config(enabled_import_methods=enabled_methods)
    response = webob.Response()
    self.serializer.create(response, self.fixtures[0])
    self.assertEqual(http.CREATED, response.status_int)
    header_value = response.headers.get(header_name)
    self.assertIsNotNone(header_value)
    self.assertCountEqual(enabled_methods, header_value.split(','))
    self.config(enabled_import_methods=['swift-party-time'])
    response = webob.Response()
    self.serializer.create(response, self.fixtures[0])
    self.assertEqual(http.CREATED, response.status_int)
    header_value = response.headers.get(header_name)
    self.assertIsNotNone(header_value)
    self.assertEqual('swift-party-time', header_value)
    self.config(enabled_import_methods=[])
    response = webob.Response()
    self.serializer.create(response, self.fixtures[0])
    self.assertEqual(http.CREATED, response.status_int)
    headers = response.headers.keys()
    self.assertNotIn(header_name, headers)