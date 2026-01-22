from unittest import mock
import webob.exc
from glance.api.v2 import policy
from glance.common import exception
from glance.tests import utils
def test_update_locations(self):
    self.policy.update_locations()
    self.enforcer.enforce.assert_called_once_with(self.context, 'set_image_location', mock.ANY)