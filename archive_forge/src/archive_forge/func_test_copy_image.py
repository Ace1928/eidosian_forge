from unittest import mock
import webob.exc
from glance.api.v2 import policy
from glance.common import exception
from glance.tests import utils
def test_copy_image(self):
    self.policy.copy_image()
    self.enforcer.enforce.assert_called_once_with(self.context, 'copy_image', mock.ANY)