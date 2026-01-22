from io import StringIO
import numpy as np
import pytest
from pandas.compat import is_platform_linux
from pandas import DataFrame
import pandas._testing as tm
@pytest.mark.skip_ubsan
@xfail_pyarrow
@pytest.mark.parametrize('exp', [999999999999999999, -999999999999999999])
def test_too_many_exponent_digits(all_parsers_all_precisions, exp, request):
    parser, precision = all_parsers_all_precisions
    data = f'data\n10E{exp}'
    result = parser.read_csv(StringIO(data), float_precision=precision)
    if precision == 'round_trip':
        if exp == 999999999999999999 and is_platform_linux():
            mark = pytest.mark.xfail(reason='GH38794, on Linux gives object result')
            request.applymarker(mark)
        value = np.inf if exp > 0 else 0.0
        expected = DataFrame({'data': [value]})
    else:
        expected = DataFrame({'data': [f'10E{exp}']})
    tm.assert_frame_equal(result, expected)