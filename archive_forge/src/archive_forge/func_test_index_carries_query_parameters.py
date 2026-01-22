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
def test_index_carries_query_parameters(self):
    url = '/v2/images?limit=10&sort_key=id&sort_dir=asc'
    request = webob.Request.blank(url)
    response = webob.Response(request=request)
    result = {'images': self.fixtures, 'next_marker': UUID2}
    self.serializer.index(response, result)
    output = jsonutils.loads(response.body)
    expected_url = '/v2/images?limit=10&sort_dir=asc&sort_key=id'
    self.assertEqual(unit_test_utils.sort_url_by_qs_keys(expected_url), unit_test_utils.sort_url_by_qs_keys(output['first']))
    expect_next = '/v2/images?limit=10&marker=%s&sort_dir=asc&sort_key=id'
    self.assertEqual(unit_test_utils.sort_url_by_qs_keys(expect_next % UUID2), unit_test_utils.sort_url_by_qs_keys(output['next']))