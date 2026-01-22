from .._util import get_backend
from ..util.regression import least_squares
from ..util.pyutil import defaultnamedtuple
from ..units import default_constants, default_units, format_string, patched_numpy
def equation_as_string(self, precision, tex=False):
    (str_A, str_A_unit), (str_Ea, str_Ea_unit) = self.format(precision, tex)
    if tex:
        return ('{}\\exp \\left(-\\frac{{{}}}{{RT}} \\right)'.format(str_A, str_Ea + ' ' + str_Ea_unit), str_A_unit)
    else:
        return ('{}*exp(-{}/(R*T))'.format(str_A, str_Ea + ' ' + str_Ea_unit), str_A_unit)