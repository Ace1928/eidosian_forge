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
def test_index_forbidden_get_image_location(self):
    """Make sure the serializer works fine.

        No matter if current user is authorized to get image location if the
        show_multiple_locations is False.

        """

    class ImageLocations(object):

        def __len__(self):
            raise exception.Forbidden()
    self.config(show_multiple_locations=False)
    self.config(show_image_direct_url=False)
    url = '/v2/images?limit=10&sort_key=id&sort_dir=asc'
    request = webob.Request.blank(url)
    response = webob.Response(request=request)
    result = {'images': self.fixtures}
    self.assertEqual(http.OK, response.status_int)
    result['images'][0].locations = ImageLocations()
    self.serializer.index(response, result)
    self.assertEqual(http.OK, response.status_int)