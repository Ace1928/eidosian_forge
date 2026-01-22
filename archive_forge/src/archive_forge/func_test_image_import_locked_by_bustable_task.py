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
@mock.patch.object(glance.notifier.ImageRepoProxy, 'save')
@mock.patch('glance.db.simple.api.image_set_property_atomic')
@mock.patch('glance.db.simple.api.image_delete_property_atomic')
@mock.patch.object(glance.notifier.TaskFactoryProxy, 'new_task')
@mock.patch.object(glance.notifier.ImageRepoProxy, 'get')
def test_image_import_locked_by_bustable_task(self, mock_get, mock_nt, mock_dpi, mock_spi, mock_save, task_status='processing'):
    if task_status == 'processing':
        task_input = {'backend': ['store2']}
    else:
        task_input = {}
    task = test_tasks_resource._db_fixture(test_tasks_resource.UUID1, status=task_status, input=task_input)
    self.db.task_create(None, task)
    image = FakeImage(status='uploading')
    image.extra_properties['os_glance_import_task'] = task['id']
    image.extra_properties['os_glance_importing_to_stores'] = 'store2'
    mock_get.return_value = image
    request = unit_test_utils.get_fake_request(tenant=TENANT1)
    req_body = {'method': {'name': 'glance-direct'}}
    time_fixture = fixture.TimeFixture(task['updated_at'] + datetime.timedelta(minutes=10))
    self.useFixture(time_fixture)
    mock_nt.return_value.task_id = 'mytask'
    self.assertRaises(webob.exc.HTTPConflict, self.controller.import_image, request, UUID1, req_body)
    mock_dpi.assert_not_called()
    mock_spi.assert_not_called()
    mock_nt.assert_not_called()
    time_fixture.advance_time_delta(datetime.timedelta(minutes=90))
    self.controller.import_image(request, UUID1, req_body)
    mock_dpi.assert_called_once_with(image.id, 'os_glance_import_task', task['id'])
    mock_spi.assert_called_once_with(image.id, 'os_glance_import_task', 'mytask')
    if task_status == 'processing':
        self.assertNotIn('store2', image.extra_properties['os_glance_importing_to_stores'])