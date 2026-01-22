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
def test_update_reserved_not_counted_in_quota(self):
    self.config(image_property_quota=1)
    request = unit_test_utils.get_fake_request()
    self.db.image_update(None, UUID1, {'properties': {'os_glance_foo': '123', 'os_glance_bar': 456}})
    changes = [{'op': 'add', 'path': ['foo'], 'value': 'baz'}]
    self.controller.update(request, UUID1, changes)
    changes = [{'op': 'add', 'path': ['snitch'], 'value': 'golden'}]
    self.assertRaises(webob.exc.HTTPRequestEntityTooLarge, self.controller.update, request, UUID1, changes)