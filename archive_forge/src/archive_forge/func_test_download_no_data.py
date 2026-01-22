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
def test_download_no_data(self):
    resp = utils.FakeResponse(headers={}, status_code=204)
    self.controller.controller.http_client.get = mock.Mock(return_value=(resp, {}))
    self.controller.data('image_id')