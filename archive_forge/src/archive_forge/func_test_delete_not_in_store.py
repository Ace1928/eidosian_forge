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
def test_delete_not_in_store(self):
    request = unit_test_utils.get_fake_request()
    self.assertIn('%s/%s' % (BASE_URI, UUID1), self.store.data)
    for k in self.store.data:
        if UUID1 in k:
            del self.store.data[k]
            break
    self.controller.delete(request, UUID1)
    deleted_img = self.db.image_get(request.context, UUID1, force_show_deleted=True)
    self.assertTrue(deleted_img['deleted'])
    self.assertEqual('deleted', deleted_img['status'])
    self.assertNotIn('%s/%s' % (BASE_URI, UUID1), self.store.data)