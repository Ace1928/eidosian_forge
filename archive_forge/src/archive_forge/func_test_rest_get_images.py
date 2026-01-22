import collections
import http.client as http
import io
from unittest import mock
import copy
import os
import sys
import uuid
import fixtures
from oslo_serialization import jsonutils
import webob
from glance.cmd import replicator as glance_replicator
from glance.common import exception
from glance.tests.unit import utils as unit_test_utils
from glance.tests import utils as test_utils
def test_rest_get_images(self):
    c = glance_replicator.ImageService(FakeHTTPConnection(), 'noauth')
    resp = {'images': [IMG_RESPONSE_ACTIVE, IMG_RESPONSE_QUEUED]}
    c.conn.prime_request('GET', 'v1/images/detail?is_public=None', '', {'x-auth-token': 'noauth'}, http.OK, jsonutils.dumps(resp), {})
    c.conn.prime_request('GET', 'v1/images/detail?marker=%s&is_public=None' % IMG_RESPONSE_QUEUED['id'], '', {'x-auth-token': 'noauth'}, http.OK, jsonutils.dumps({'images': []}), {})
    imgs = list(c.get_images())
    self.assertEqual(2, len(imgs))
    self.assertEqual(2, c.conn.count)