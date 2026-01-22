from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('usecols,expected', [(lambda x: x.upper() in ['AAA', 'BBB', 'DDD'], DataFrame({'AaA': {0: 0.056674973, 1: 2.6132309819999997, 2: 3.5689350380000002}, 'bBb': {0: 8, 1: 2, 2: 7}, 'ddd': {0: 'a', 1: 'b', 2: 'a'}})), (lambda x: False, DataFrame(columns=Index([])))])
def test_callable_usecols(all_parsers, usecols, expected):
    data = 'AaA,bBb,CCC,ddd\n0.056674973,8,True,a\n2.613230982,2,False,b\n3.568935038,7,False,a'
    parser = all_parsers
    if parser.engine == 'pyarrow':
        msg = "The pyarrow engine does not allow 'usecols' to be a callable"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), usecols=usecols)
        return
    result = parser.read_csv(StringIO(data), usecols=usecols)
    tm.assert_frame_equal(result, expected)