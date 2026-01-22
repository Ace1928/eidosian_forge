from unittest import mock
import netaddr
import testtools
from neutron_lib.api import converters
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
from neutron_lib.tests import tools
def test_convert_kvp_list_to_dict_succeeds_for_values(self):
    result = converters.convert_kvp_list_to_dict(['a=b', 'c=d'])
    self.assertEqual({'a': ['b'], 'c': ['d']}, result)