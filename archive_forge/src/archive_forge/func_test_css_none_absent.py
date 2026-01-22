import pytest
from pandas.errors import CSSWarning
import pandas._testing as tm
from pandas.io.formats.css import CSSResolver
@pytest.mark.parametrize('style,equiv', [('margin: 1px; margin-top: inherit', 'margin-bottom: 1px; margin-right: 1px; margin-left: 1px'), ('margin-top: inherit', ''), ('margin-top: initial', '')])
def test_css_none_absent(style, equiv):
    assert_same_resolution(style, equiv)