import hashlib
from unittest import mock
from oslo_utils import encodeutils
from oslo_utils import excutils
from oslo_utils import uuidutils
from neutron_lib.api.definitions import portbindings_extended as pb_ext
from neutron_lib import constants
from neutron_lib import exceptions
from neutron_lib.plugins import utils
from neutron_lib.tests import _base as base
def test_parse_network_vlan_ranges(self):
    ranges = utils.parse_network_vlan_ranges(['n1:1:3', 'n2:2:4', 'n3', 'n4', 'n4:10:12'])
    self.assertEqual(4, len(ranges.keys()))
    self.assertIn('n1', ranges.keys())
    self.assertIn('n2', ranges.keys())
    self.assertEqual(2, len(ranges['n1'][0]))
    self.assertEqual(1, ranges['n1'][0][0])
    self.assertEqual(3, ranges['n1'][0][1])
    self.assertEqual(2, len(ranges['n2'][0]))
    self.assertEqual(2, ranges['n2'][0][0])
    self.assertEqual(4, ranges['n2'][0][1])
    self.assertEqual([constants.VLAN_VALID_RANGE], ranges['n3'])
    self.assertEqual([constants.VLAN_VALID_RANGE], ranges['n4'])