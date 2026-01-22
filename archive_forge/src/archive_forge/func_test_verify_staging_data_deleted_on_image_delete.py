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
@mock.patch.object(store, 'get_store_from_store_identifier')
@mock.patch.object(store.location, 'get_location_from_uri_and_backend')
@mock.patch.object(store_utils, 'get_dir_separator')
def test_verify_staging_data_deleted_on_image_delete(self, mock_get_dir_separator, mock_location, mock_store):
    self.config(enabled_backends={'fake-store': 'file'})
    fake_staging_store = mock.Mock()
    mock_store.return_value = fake_staging_store
    mock_get_dir_separator.return_value = ('/', '/tmp/os_glance_staging_store')
    image_id = str(uuid.uuid4())
    self.images = [_db_fixture(image_id, owner=TENANT1, name='1', disk_format='raw', container_format='bare', status='importing', checksum=None, os_hash_algo=None, os_hash_value=None)]
    self.db.image_create(None, self.images[0])
    request = unit_test_utils.get_fake_request()
    try:
        self.controller.delete(request, image_id)
        self.assertEqual(1, mock_store.call_count)
        mock_store.assert_called_once_with('os_glance_staging_store')
        self.assertEqual(1, mock_location.call_count)
        fake_staging_store.delete.assert_called_once()
    except Exception as e:
        self.fail('Delete raised exception: %s' % e)
    deleted_img = self.db.image_get(request.context, image_id, force_show_deleted=True)
    self.assertTrue(deleted_img['deleted'])
    self.assertEqual('deleted', deleted_img['status'])