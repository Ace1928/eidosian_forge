from unittest import mock
from glance.api import property_protections
from glance import context
from glance import gateway
from glance import notifier
from glance import quota
from glance.tests.unit import utils as unit_test_utils
import glance.tests.utils as test_utils
@mock.patch('glance.common.property_utils.PropertyRules._load_rules')
def test_get_image_factory_with_pp(self, mock_load):
    self.config(property_protection_file='foo')
    factory = self.gateway.get_image_factory(self.context)
    self.assertIsInstance(factory, property_protections.ProtectedImageFactoryProxy)