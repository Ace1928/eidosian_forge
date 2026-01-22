from unittest import mock
import webob.exc
from glance.api.v2 import policy
from glance.common import exception
from glance.tests import utils
def test_add_image_translates_owner_failure(self):
    self.policy = policy.ImageAPIPolicy(self.context, {'owner': 'else'}, enforcer=self.enforcer)
    self.policy.add_image()
    self.enforcer.enforce.side_effect = exception.Duplicate
    self.assertRaises(exception.Duplicate, self.policy.add_image)
    self.enforcer.enforce.side_effect = webob.exc.HTTPForbidden('original')
    exc = self.assertRaises(webob.exc.HTTPForbidden, self.policy.add_image)
    self.assertIn('You are not permitted to create images owned by', str(exc))
    self.policy = policy.ImageAPIPolicy(self.context, {}, enforcer=self.enforcer)
    exc = self.assertRaises(webob.exc.HTTPForbidden, self.policy.add_image)
    self.assertIn('original', str(exc))