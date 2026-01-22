from unittest import mock
import webob.exc
from glance.api.v2 import policy
from glance.common import exception
from glance.tests import utils
def test_deactivate_image(self):
    self.policy.deactivate_image()
    self.enforcer.enforce.assert_called_once_with(self.context, 'deactivate', mock.ANY)