import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
def test_string_to_bytes(self):

    def _get_quantity(sign, magnitude, unit_suffix):
        res = float('%s%s' % (sign, magnitude))
        if unit_suffix in ['b', 'bit']:
            res /= 8
        return res

    def _get_constant(unit_prefix, unit_system):
        if not unit_prefix:
            return 1
        elif unit_system == 'SI':
            res = getattr(units, unit_prefix)
        elif unit_system == 'IEC':
            if unit_prefix.endswith('i'):
                res = getattr(units, unit_prefix)
            else:
                res = getattr(units, '%si' % unit_prefix)
        elif unit_system == 'mixed':
            if unit_prefix == 'K':
                unit_prefix = 'k'
            res = getattr(units, unit_prefix)
        return res
    text = ''.join([self.sign, self.magnitude, self.unit_prefix, self.unit_suffix])
    err_si = self.unit_system == 'SI' and (self.unit_prefix == 'K' or self.unit_prefix.endswith('i'))
    err_iec = self.unit_system == 'IEC' and self.unit_prefix == 'k'
    if getattr(self, 'assert_error', False) or err_si or err_iec:
        self.assertRaises(ValueError, strutils.string_to_bytes, text, unit_system=self.unit_system, return_int=self.return_int)
        return
    quantity = _get_quantity(self.sign, self.magnitude, self.unit_suffix)
    constant = _get_constant(self.unit_prefix, self.unit_system)
    expected = quantity * constant
    actual = strutils.string_to_bytes(text, unit_system=self.unit_system, return_int=self.return_int)
    if self.return_int:
        self.assertEqual(actual, int(math.ceil(expected)))
    else:
        self.assertAlmostEqual(actual, expected)