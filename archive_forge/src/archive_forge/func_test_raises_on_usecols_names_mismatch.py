from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('usecols,kwargs,expected,msg', [(['a', 'b', 'c', 'd'], {}, DataFrame({'a': [1, 5], 'b': [2, 6], 'c': [3, 7], 'd': [4, 8]}), None), (['a', 'b', 'c', 'f'], {}, None, _msg_validate_usecols_names.format("\\['f'\\]")), (['a', 'b', 'f'], {}, None, _msg_validate_usecols_names.format("\\['f'\\]")), (['a', 'b', 'f', 'g'], {}, None, _msg_validate_usecols_names.format("\\[('f', 'g'|'g', 'f')\\]")), (None, {'header': 0, 'names': ['A', 'B', 'C', 'D']}, DataFrame({'A': [1, 5], 'B': [2, 6], 'C': [3, 7], 'D': [4, 8]}), None), (['A', 'B', 'C', 'f'], {'header': 0, 'names': ['A', 'B', 'C', 'D']}, None, _msg_validate_usecols_names.format("\\['f'\\]")), (['A', 'B', 'f'], {'names': ['A', 'B', 'C', 'D']}, None, _msg_validate_usecols_names.format("\\['f'\\]"))])
def test_raises_on_usecols_names_mismatch(all_parsers, usecols, kwargs, expected, msg, request):
    data = 'a,b,c,d\n1,2,3,4\n5,6,7,8'
    kwargs.update(usecols=usecols)
    parser = all_parsers
    if parser.engine == 'pyarrow' and (not (usecols is not None and expected is not None)):
        pytest.skip(reason='https://github.com/apache/arrow/issues/38676')
    if expected is None:
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), **kwargs)
    else:
        result = parser.read_csv(StringIO(data), **kwargs)
        tm.assert_frame_equal(result, expected)