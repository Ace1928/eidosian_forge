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
@mock.patch.object(glance.notifier.ImageRepoProxy, 'get')
def test_image_import_locked(self, mock_get):
    task = test_tasks_resource._db_fixture(test_tasks_resource.UUID1, status='pending')
    self.db.task_create(None, task)
    image = FakeImage(status='uploading')
    image.extra_properties['os_glance_import_task'] = task['id']
    mock_get.return_value = image
    request = unit_test_utils.get_fake_request(tenant=TENANT1)
    req_body = {'method': {'name': 'glance-direct'}}
    exc = self.assertRaises(webob.exc.HTTPConflict, self.controller.import_image, request, UUID1, req_body)
    self.assertEqual('Image has active task', str(exc))