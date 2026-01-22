from unittest import mock
import webob.exc
from glance.api.v2 import policy
from glance.common import exception
from glance.tests import utils
def test_delete_locations_falls_back_to_legacy(self):
    self.config(enforce_new_defaults=False, group='oslo_policy')
    self.config(enforce_scope=False, group='oslo_policy')
    self.context.is_admin = True
    self.context.owner = 'someuser'
    self.image.owner = 'someotheruser'
    self.policy.delete_locations()
    self.context.is_admin = False
    self.context.owner = 'someuser'
    self.image.owner = 'someuser'
    self.policy.delete_locations()
    self.image.owner = 'someotheruser'
    self.assertRaises(exception.Forbidden, self.policy.delete_locations)
    with mock.patch('glance.api.v2.policy.check_is_image_mutable') as m:
        self.policy.delete_locations()
        m.assert_called_once_with(self.context, self.image)
    self.config(enforce_new_defaults=True, group='oslo_policy')
    self.config(enforce_scope=True, group='oslo_policy')
    with mock.patch('glance.api.v2.policy.check_is_image_mutable') as m:
        self.policy.delete_locations()
        self.assertFalse(m.called)