from io import BytesIO
import pytest
import logging
from matplotlib import _afm
from matplotlib import font_manager as fm
@pytest.mark.parametrize('afm_data', [b'StartFontMetrics 2.0\nComment Comments are ignored.\nComment Creation Date:Mon Nov 13 12:34:11 GMT 2017\nAardvark bob\nFontName MyFont-Bold\nEncodingScheme FontSpecific\nStartCharMetrics 3', b'StartFontMetrics 2.0\nComment Comments are ignored.\nComment Creation Date:Mon Nov 13 12:34:11 GMT 2017\nItalicAngle zero degrees\nFontName MyFont-Bold\nEncodingScheme FontSpecific\nStartCharMetrics 3'])
def test_malformed_header(afm_data, caplog):
    fh = BytesIO(afm_data)
    with caplog.at_level(logging.ERROR):
        _afm._parse_header(fh)
    assert len(caplog.records) == 1