from unittest import mock
import uuid
from neutron_lib.placement import utils as place_utils
from neutron_lib.tests import _base as base
def test_parse_rp_options(self):
    self.assertEqual({}, place_utils._parse_rp_options([], tuple()))
    self.assertEqual({'outer_key0': {'inner_key0': None}}, place_utils._parse_rp_options(['outer_key0'], ('inner_key0',)))
    self.assertEqual({'outer_key0': {'inner_key0': None}, 'outer_key1': {'inner_key0': None}, 'outer_key2': {'inner_key0': None}, 'outer_key3': {'inner_key0': None}}, place_utils._parse_rp_options(['outer_key0', 'outer_key1', 'outer_key2', 'outer_key3'], ('inner_key0',)))
    self.assertEqual({'outer_key0': {'inner_key0': None, 'inner_key1': None, 'inner_key2': None, 'inner_key3': None}}, place_utils._parse_rp_options(['outer_key0'], ('inner_key0', 'inner_key1', 'inner_key2', 'inner_key3')))
    self.assertEqual({'outer_key0': {'inner_key0': None, 'inner_key1': None, 'inner_key2': None, 'inner_key3': None}, 'outer_key1': {'inner_key0': None, 'inner_key1': None, 'inner_key2': None, 'inner_key3': None}, '': {'inner_key0': None, 'inner_key1': None, 'inner_key2': None, 'inner_key3': None}, 'outer_key3': {'inner_key0': None, 'inner_key1': 1, 'inner_key2': 2, 'inner_key3': 3}}, place_utils._parse_rp_options(['outer_key0', 'outer_key1::::', '::::', 'outer_key3::1:2:3'], ('inner_key0', 'inner_key1', 'inner_key2', 'inner_key3')))
    self.assertRaises(ValueError, place_utils._parse_rp_options, ['outer_key0:'], ('inner_key1', 'inner_key2'))
    self.assertRaises(ValueError, place_utils._parse_rp_options, ['outer_key0::'], ('inner_key1',))