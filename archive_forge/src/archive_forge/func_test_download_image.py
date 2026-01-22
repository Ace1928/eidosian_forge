from unittest import mock
import webob.exc
from glance.api.v2 import policy
from glance.common import exception
from glance.tests import utils
def test_download_image(self):
    self.policy.download_image()
    self.enforcer.enforce.assert_called_once_with(self.context, 'download_image', mock.ANY)