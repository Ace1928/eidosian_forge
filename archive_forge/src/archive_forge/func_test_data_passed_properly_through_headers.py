import datetime
import gettext
import http.client as http
import os
import socket
from unittest import mock
import eventlet.patcher
import fixtures
from oslo_concurrency import processutils
from oslo_serialization import jsonutils
import routes
import webob
from glance.api.v2 import router as router_v2
from glance.common import exception
from glance.common import utils
from glance.common import wsgi
from glance import i18n
from glance.image_cache import prefetcher
from glance.tests import utils as test_utils
def test_data_passed_properly_through_headers(self):
    """
        Verifies that data is the same after being passed through headers
        """
    fixture = {'is_public': True, 'deleted': False, 'name': None, 'size': 19, 'location': 'file:///tmp/glance-tests/2', 'properties': {'distro': 'Ubuntu 10.04 LTS'}}
    headers = utils.image_meta_to_http_headers(fixture)

    class FakeResponse(object):
        pass
    response = FakeResponse()
    response.headers = headers
    result = utils.get_image_meta_from_headers(response)
    for k, v in fixture.items():
        if v is not None:
            self.assertEqual(v, result[k])
        else:
            self.assertNotIn(k, result)