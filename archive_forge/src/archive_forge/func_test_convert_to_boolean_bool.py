from unittest import mock
import netaddr
import testtools
from neutron_lib.api import converters
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
from neutron_lib.tests import tools
def test_convert_to_boolean_bool(self):
    self.assertIs(converters.convert_to_boolean(True), True)
    self.assertIs(converters.convert_to_boolean(False), False)