from io import StringIO
import pytest
from pandas import DataFrame
import pandas._testing as tm
@pytest.mark.parametrize('data,names,expected', [('a,b,b\n1,2,3', ['a.1', 'a.1', 'a.1.1'], DataFrame([['a', 'b', 'b'], ['1', '2', '3']], columns=['a.1', 'a.1.1', 'a.1.1.1'])), ('a,b,c,d,e,f\n1,2,3,4,5,6', ['a', 'a', 'a.1', 'a.1.1', 'a.1.1.1', 'a.1.1.1.1'], DataFrame([['a', 'b', 'c', 'd', 'e', 'f'], ['1', '2', '3', '4', '5', '6']], columns=['a', 'a.1', 'a.1.1', 'a.1.1.1', 'a.1.1.1.1', 'a.1.1.1.1.1'])), ('a,b,c,d,e,f,g\n1,2,3,4,5,6,7', ['a', 'a', 'a.3', 'a.1', 'a.2', 'a', 'a'], DataFrame([['a', 'b', 'c', 'd', 'e', 'f', 'g'], ['1', '2', '3', '4', '5', '6', '7']], columns=['a', 'a.1', 'a.3', 'a.1.1', 'a.2', 'a.2.1', 'a.3.1']))])
def test_thorough_mangle_names(all_parsers, data, names, expected):
    parser = all_parsers
    with pytest.raises(ValueError, match='Duplicate names'):
        parser.read_csv(StringIO(data), names=names)