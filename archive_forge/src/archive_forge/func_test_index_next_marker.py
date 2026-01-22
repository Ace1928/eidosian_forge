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
def test_index_next_marker(self):
    request = webob.Request.blank('/v2/images')
    response = webob.Response(request=request)
    result = {'images': self.fixtures, 'next_marker': UUID2}
    self.serializer.index(response, result)
    output = jsonutils.loads(response.body)
    self.assertEqual('/v2/images?marker=%s' % UUID2, output['next'])