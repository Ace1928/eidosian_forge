from io import StringIO
import pytest
from pandas import DataFrame
import pandas._testing as tm
@pytest.mark.parametrize('usecols', [['あああ', 'いい'], ['あああ', 'いい']])
def test_usecols_with_multi_byte_characters(all_parsers, usecols):
    data = 'あああ,いい,ううう,ええええ\n0.056674973,8,True,a\n2.613230982,2,False,b\n3.568935038,7,False,a'
    parser = all_parsers
    exp_data = {'あああ': {0: 0.056674973, 1: 2.6132309819999997, 2: 3.5689350380000002}, 'いい': {0: 8, 1: 2, 2: 7}}
    expected = DataFrame(exp_data)
    result = parser.read_csv(StringIO(data), usecols=usecols)
    tm.assert_frame_equal(result, expected)