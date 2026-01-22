import errno
import hashlib
import testtools
from unittest import mock
import ddt
from glanceclient.common import utils as common_utils
from glanceclient import exc
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import images
def test_update_bad_additionalProperty_type(self):
    image_id = 'e7e59ff6-fa2e-4075-87d3-1a1398a07dc3'
    params = {'name': 'pong', 'bad_prop': False}
    with testtools.ExpectedException(TypeError):
        self.controller.update(image_id, **params)