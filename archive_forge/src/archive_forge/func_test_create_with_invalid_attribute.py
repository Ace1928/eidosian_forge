import copy
from unittest import mock
import testtools
from ironicclient.common import base
from ironicclient import exc
from ironicclient.tests.unit import utils
def test_create_with_invalid_attribute(self):
    self.assertRaisesRegex(exc.InvalidAttribute, 'non-existent-attribute', self.manager.create, **INVALID_ATTRIBUTE_TESTABLE_RESOURCE)