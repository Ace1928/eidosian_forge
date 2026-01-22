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
def test_update_reserved_attributes(self):
    samples = {'deleted': False, 'deleted_at': ISOTIME, 'os_glance_import_task': 'foo', 'os_glance_anything': 'bar', 'os_glance_': 'baz', 'os_glance': 'bat'}
    for key, value in samples.items():
        request = self._get_fake_patch_request()
        body = [{'op': 'replace', 'path': '/%s' % key, 'value': value}]
        request.body = jsonutils.dump_as_bytes(body)
        try:
            self.deserializer.update(request)
        except webob.exc.HTTPForbidden:
            pass
        else:
            self.fail('Updating %s did not result in HTTPForbidden' % key)