from io import StringIO
import pytest
import pandas._testing as tm
def test_verbose_read2(all_parsers, capsys):
    parser = all_parsers
    data = 'a,b,c,d\none,1,2,3\ntwo,1,2,3\nthree,1,2,3\nfour,1,2,3\nfive,1,2,3\n,1,2,3\nseven,1,2,3\neight,1,2,3'
    if parser.engine == 'pyarrow':
        msg = "The 'verbose' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=depr_msg, check_stacklevel=False):
                parser.read_csv(StringIO(data), verbose=True, index_col=0)
        return
    with tm.assert_produces_warning(FutureWarning, match=depr_msg, check_stacklevel=False):
        parser.read_csv(StringIO(data), verbose=True, index_col=0)
    captured = capsys.readouterr()
    if parser.engine == 'c':
        assert 'Tokenization took:' in captured.out
        assert 'Parser memory cleanup took:' in captured.out
    else:
        assert captured.out == 'Filled 1 NA values in column a\n'