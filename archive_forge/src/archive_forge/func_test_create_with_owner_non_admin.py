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
def test_create_with_owner_non_admin(self):
    enforcer = unit_test_utils.enforcer_from_rules({'add_image': 'role:member,reader'})
    request = unit_test_utils.get_fake_request()
    request.context.is_admin = False
    image = {'owner': '12345'}
    self.controller.policy = enforcer
    self.assertRaises(webob.exc.HTTPForbidden, self.controller.create, request, image=image, extra_properties={}, tags=[])
    enforcer = unit_test_utils.enforcer_from_rules({'add_image': "'{0}':%(owner)s".format(TENANT1)})
    request = unit_test_utils.get_fake_request()
    request.context.is_admin = False
    image = {'owner': TENANT1}
    self.controller.policy = enforcer
    output = self.controller.create(request, image=image, extra_properties={}, tags=[])
    self.assertEqual(TENANT1, output.owner)