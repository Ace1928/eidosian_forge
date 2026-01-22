from io import (
import os
import platform
from urllib.error import URLError
import uuid
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data,kwargs,expected,msg', [('a,b,c\n4,5,6\n ', {}, DataFrame([[4, 5, 6]], columns=['a', 'b', 'c']), None), ('a,b,c\n4,5,6\n#comment', {'comment': '#'}, DataFrame([[4, 5, 6]], columns=['a', 'b', 'c']), None), ('a,b,c\n4,5,6\n\r', {}, DataFrame([[4, 5, 6]], columns=['a', 'b', 'c']), None), ('a,b,c\n4,5,6#comment', {'comment': '#'}, DataFrame([[4, 5, 6]], columns=['a', 'b', 'c']), None), ('a,b,c\n4,5,6\nskipme', {'skiprows': [2]}, DataFrame([[4, 5, 6]], columns=['a', 'b', 'c']), None), ('a,b,c\n4,5,6\n#comment', {'comment': '#', 'skip_blank_lines': False}, DataFrame([[4, 5, 6]], columns=['a', 'b', 'c']), None), ('a,b,c\n4,5,6\n ', {'skip_blank_lines': False}, DataFrame([['4', 5, 6], [' ', None, None]], columns=['a', 'b', 'c']), None), ('a,b,c\n4,5,6\n\r', {'skip_blank_lines': False}, DataFrame([[4, 5, 6], [None, None, None]], columns=['a', 'b', 'c']), None), ('a,b,c\n4,5,6\n\\', {'escapechar': '\\'}, None, '(EOF following escape character)|(unexpected end of data)'), ('a,b,c\n4,5,6\n"\\', {'escapechar': '\\'}, None, '(EOF inside string starting at row 2)|(unexpected end of data)'), ('a,b,c\n4,5,6\n"', {'escapechar': '\\'}, None, '(EOF inside string starting at row 2)|(unexpected end of data)')], ids=['whitespace-line', 'eat-line-comment', 'eat-crnl-nop', 'eat-comment', 'skip-line', 'eat-line-comment', 'in-field', 'eat-crnl', 'escaped-char', 'escape-in-quoted-field', 'in-quoted-field'])
def test_eof_states(all_parsers, data, kwargs, expected, msg, request):
    parser = all_parsers
    if parser.engine == 'pyarrow' and 'comment' in kwargs:
        msg = "The 'comment' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), **kwargs)
        return
    if parser.engine == 'pyarrow' and '\r' not in data:
        pytest.skip(reason='https://github.com/apache/arrow/issues/38676')
    if expected is None:
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data), **kwargs)
    else:
        result = parser.read_csv(StringIO(data), **kwargs)
        tm.assert_frame_equal(result, expected)