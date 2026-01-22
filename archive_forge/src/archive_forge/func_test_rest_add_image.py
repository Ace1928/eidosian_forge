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
def test_rest_add_image(self):
    c = glance_replicator.ImageService(FakeHTTPConnection(), 'noauth')
    image_body = 'THISISANIMAGEBODYFORSURE!'
    image_meta_with_proto = {'x-auth-token': 'noauth', 'Content-Type': 'application/octet-stream', 'Content-Length': len(image_body)}
    for key in IMG_RESPONSE_ACTIVE:
        image_meta_with_proto['x-image-meta-%s' % key] = IMG_RESPONSE_ACTIVE[key]
    c.conn.prime_request('POST', 'v1/images', image_body, image_meta_with_proto, http.OK, '', IMG_RESPONSE_ACTIVE)
    headers, body = c.add_image(IMG_RESPONSE_ACTIVE, image_body)
    self.assertEqual(IMG_RESPONSE_ACTIVE, headers)
    self.assertEqual(1, c.conn.count)