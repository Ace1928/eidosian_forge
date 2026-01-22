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
def test_index_with_non_default_is_public_filter(self):
    private_uuid = str(uuid.uuid4())
    new_image = _db_fixture(private_uuid, visibility='private', owner=TENANT3)
    self.db.image_create(None, new_image)
    path = '/images?visibility=private'
    request = unit_test_utils.get_fake_request(path, is_admin=True)
    output = self.controller.index(request, filters={'visibility': 'private'})
    self.assertEqual(1, len(output['images']))
    actual = set([image.image_id for image in output['images']])
    expected = set([private_uuid])
    self.assertEqual(expected, actual)
    path = '/images?visibility=shared'
    request = unit_test_utils.get_fake_request(path, is_admin=True)
    output = self.controller.index(request, filters={'visibility': 'shared'})
    self.assertEqual(1, len(output['images']))
    actual = set([image.image_id for image in output['images']])
    expected = set([UUID4])
    self.assertEqual(expected, actual)