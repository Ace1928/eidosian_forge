import csv
from io import StringIO
import pytest
from pandas.errors import ParserWarning
from pandas import DataFrame
import pandas._testing as tm
def test_dialect(all_parsers):
    parser = all_parsers
    data = 'label1,label2,label3\nindex1,"a,c,e\nindex2,b,d,f\n'
    dia = csv.excel()
    dia.quoting = csv.QUOTE_NONE
    if parser.engine == 'pyarrow':
        msg = "The 'dialect' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), dialect=dia)
        return
    df = parser.read_csv(StringIO(data), dialect=dia)
    data = 'label1,label2,label3\nindex1,a,c,e\nindex2,b,d,f\n'
    exp = parser.read_csv(StringIO(data))
    exp.replace('a', '"a', inplace=True)
    tm.assert_frame_equal(df, exp)