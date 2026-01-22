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
def test_create_with_properties(self):
    request = unit_test_utils.get_fake_request()
    image_properties = {'foo': 'bar'}
    image = {'name': 'image-1'}
    output = self.controller.create(request, image=image, extra_properties=image_properties, tags=[])
    self.assertEqual('image-1', output.name)
    self.assertEqual(image_properties, output.extra_properties)
    self.assertEqual(set([]), output.tags)
    self.assertEqual('shared', output.visibility)
    output_logs = self.notifier.get_logs()
    self.assertEqual(1, len(output_logs))
    output_log = output_logs[0]
    self.assertEqual('INFO', output_log['notification_type'])
    self.assertEqual('image.create', output_log['event_type'])
    self.assertEqual('image-1', output_log['payload']['name'])