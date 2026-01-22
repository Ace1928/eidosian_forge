from unittest import mock
import uuid
from neutron_lib.placement import utils as place_utils
from neutron_lib.tests import _base as base
def test_parse_rp_pp_without_direction(self):
    self.assertEqual({}, place_utils.parse_rp_pp_without_direction([], 'default_host'))
    self.assertEqual({'host0': {'any': None}}, place_utils.parse_rp_pp_without_direction(['host0'], 'default_host'))
    self.assertEqual({'host0': {'any': None}}, place_utils.parse_rp_pp_without_direction(['host0:'], 'default_host'))
    self.assertEqual({'host0': {'any': 1}}, place_utils.parse_rp_pp_without_direction(['host0:1'], 'default_host'))
    self.assertEqual({'default_host': {'any': None}}, place_utils.parse_rp_pp_without_direction([':'], 'default_host'))
    self.assertEqual({'default_host': {'any': 0}}, place_utils.parse_rp_pp_without_direction([':0'], 'default_host'))
    self.assertEqual({'host0': {'any': 1}, 'host1': {'any': 10}}, place_utils.parse_rp_pp_without_direction(['host0:1', 'host1:10'], 'default_host'))
    self.assertRaises(ValueError, place_utils.parse_rp_pp_without_direction, ['default_host:', ':'], 'default_host')
    self.assertRaises(ValueError, place_utils.parse_rp_pp_without_direction, ['host0:', 'host0:'], 'default_host')
    self.assertRaises(ValueError, place_utils.parse_rp_pp_without_direction, ['host0:not a number'], 'default_host')