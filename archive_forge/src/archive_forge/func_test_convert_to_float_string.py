from unittest import mock
import netaddr
import testtools
from neutron_lib.api import converters
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
from neutron_lib.tests import tools
def test_convert_to_float_string(self):
    self.assertEqual(4, converters.convert_to_positive_float_or_none('4'))
    self.assertEqual(4.44, converters.convert_to_positive_float_or_none('4.44'))
    self.assertRaises(n_exc.InvalidInput, converters.convert_to_positive_float_or_none, 'garbage')