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
def test_rest_errors(self):
    c = glance_replicator.ImageService(FakeHTTPConnection(), 'noauth')
    for code, exc in [(http.BAD_REQUEST, webob.exc.HTTPBadRequest), (http.UNAUTHORIZED, webob.exc.HTTPUnauthorized), (http.FORBIDDEN, webob.exc.HTTPForbidden), (http.CONFLICT, webob.exc.HTTPConflict), (http.INTERNAL_SERVER_ERROR, webob.exc.HTTPInternalServerError)]:
        c.conn.prime_request('GET', 'v1/images/5dcddce0-cba5-4f18-9cf4-9853c7b207a6', '', {'x-auth-token': 'noauth'}, code, '', {})
        self.assertRaises(exc, c.get_image, '5dcddce0-cba5-4f18-9cf4-9853c7b207a6')