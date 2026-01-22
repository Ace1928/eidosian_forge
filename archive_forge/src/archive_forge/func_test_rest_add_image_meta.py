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
def test_rest_add_image_meta(self):
    c = glance_replicator.ImageService(FakeHTTPConnection(), 'noauth')
    image_meta = {'id': '5dcddce0-cba5-4f18-9cf4-9853c7b207a6'}
    image_meta_headers = glance_replicator.ImageService._dict_to_headers(image_meta)
    image_meta_headers['x-auth-token'] = 'noauth'
    image_meta_headers['Content-Type'] = 'application/octet-stream'
    c.conn.prime_request('PUT', 'v1/images/%s' % image_meta['id'], '', image_meta_headers, http.OK, '', '')
    headers, body = c.add_image_meta(image_meta)