from unittest import mock
import netaddr
import testtools
from neutron_lib.api import converters
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
from neutron_lib.tests import tools
def test_convert_to_empty_list(self):
    for item in (None, [], (), {}):
        self.assertEqual([], converters.convert_to_list(item))