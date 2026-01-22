import pytest
from pandas.errors import CSSWarning
import pandas._testing as tm
from pandas.io.formats.css import CSSResolver
@pytest.mark.parametrize('size,resolved', [('xx-small', '6pt'), ('x-small', f'{7.5:f}pt'), ('small', f'{9.6:f}pt'), ('medium', '12pt'), ('large', f'{13.5:f}pt'), ('x-large', '18pt'), ('xx-large', '24pt'), ('8px', '6pt'), ('1.25pc', '15pt'), ('.25in', '18pt'), ('02.54cm', '72pt'), ('25.4mm', '72pt'), ('101.6q', '72pt'), ('101.6q', '72pt')])
@pytest.mark.parametrize('relative_to', [None, '16pt'])
def test_css_absolute_font_size(size, relative_to, resolved):
    if relative_to is None:
        inherited = None
    else:
        inherited = {'font-size': relative_to}
    assert_resolves(f'font-size: {size}', {'font-size': resolved}, inherited=inherited)