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
@mock.patch('glance.db.simple.api.image_set_property_atomic')
@mock.patch('glance.api.common.get_thread_pool')
def test_image_import_raises_bad_request(self, mock_gpt, mock_spa):
    request = unit_test_utils.get_fake_request()
    with mock.patch.object(glance.notifier.ImageRepoProxy, 'get') as mock_get:
        mock_get.return_value = FakeImage(status='uploading')
        mock_gpt.return_value.spawn.side_effect = ValueError
        self.assertRaises(webob.exc.HTTPBadRequest, self.controller.import_image, request, UUID4, {'method': {'name': 'glance-direct'}})
        self.assertTrue(mock_gpt.return_value.spawn.called)