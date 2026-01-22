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
def test_delete_no_encryption_key_id(self):
    request = unit_test_utils.get_fake_request()
    extra_props = {'cinder_encryption_key_deletion_policy': 'on_image_deletion'}
    created_image = self.controller.create(request, image={'name': 'image-1'}, extra_properties=extra_props, tags=[])
    image_id = created_image.image_id
    self.controller.delete(request, image_id)
    image = self.db.image_get(request.context, image_id, force_show_deleted=True)
    self.assertTrue(image['deleted'])
    self.assertEqual('deleted', image['status'])