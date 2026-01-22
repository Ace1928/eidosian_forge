import unittest
import aniso8601
from aniso8601.duration import (
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import DurationResolution
from aniso8601.tests.compat import mock
def test_parse_duration_prescribed(self):
    testtuples = (('P1Y2M3DT4H54M6S', {'PnY': '1', 'PnM': '2', 'PnW': None, 'PnD': '3', 'TnH': '4', 'TnM': '54', 'TnS': '6'}), ('P1Y2M3DT4H54M6,5S', {'PnY': '1', 'PnM': '2', 'PnW': None, 'PnD': '3', 'TnH': '4', 'TnM': '54', 'TnS': '6.5'}), ('P1Y2M3DT4H54M6.5S', {'PnY': '1', 'PnM': '2', 'PnW': None, 'PnD': '3', 'TnH': '4', 'TnM': '54', 'TnS': '6.5'}), ('PT4H54M6,5S', {'PnY': None, 'PnM': None, 'PnW': None, 'PnD': None, 'TnH': '4', 'TnM': '54', 'TnS': '6.5'}), ('PT4H54M6.5S', {'PnY': None, 'PnM': None, 'PnW': None, 'PnD': None, 'TnH': '4', 'TnM': '54', 'TnS': '6.5'}), ('P1Y2M3D', {'PnY': '1', 'PnM': '2', 'PnW': None, 'PnD': '3'}), ('P1Y2M3,5D', {'PnY': '1', 'PnM': '2', 'PnW': None, 'PnD': '3.5'}), ('P1Y2M3.5D', {'PnY': '1', 'PnM': '2', 'PnW': None, 'PnD': '3.5'}), ('P1Y2M', {'PnY': '1', 'PnM': '2', 'PnW': None, 'PnD': None}), ('P1Y', {'PnY': '1', 'PnM': None, 'PnW': None, 'PnD': None}), ('P1,5Y', {'PnY': '1.5', 'PnM': None, 'PnW': None, 'PnD': None}), ('P1.5Y', {'PnY': '1.5', 'PnM': None, 'PnW': None, 'PnD': None}), ('P1M', {'PnY': None, 'PnM': '1', 'PnW': None, 'PnD': None}), ('P1,5M', {'PnY': None, 'PnM': '1.5', 'PnW': None, 'PnD': None}), ('P1.5M', {'PnY': None, 'PnM': '1.5', 'PnW': None, 'PnD': None}), ('P1W', {'PnY': None, 'PnM': None, 'PnW': '1', 'PnD': None}), ('P1,5W', {'PnY': None, 'PnM': None, 'PnW': '1.5', 'PnD': None}), ('P1.5W', {'PnY': None, 'PnM': None, 'PnW': '1.5', 'PnD': None}), ('P1D', {'PnY': None, 'PnM': None, 'PnW': None, 'PnD': '1'}), ('P1,5D', {'PnY': None, 'PnM': None, 'PnW': None, 'PnD': '1.5'}), ('P1.5D', {'PnY': None, 'PnM': None, 'PnW': None, 'PnD': '1.5'}))
    for testtuple in testtuples:
        result = _parse_duration_prescribed(testtuple[0])
        self.assertEqual(result, testtuple[1])