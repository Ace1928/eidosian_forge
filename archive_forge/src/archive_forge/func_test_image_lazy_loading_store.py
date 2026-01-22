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
def test_image_lazy_loading_store(self):
    existing_image = self.images[1]
    self.assertNotIn('store', existing_image['locations'][0]['metadata'])
    request = unit_test_utils.get_fake_request()
    with mock.patch.object(store_utils, '_get_store_id_from_uri') as mock_uri:
        mock_uri.return_value = 'fast'
        image = self.controller.show(request, UUID2)
        for loc in image.locations:
            self.assertIn('store', loc['metadata'])