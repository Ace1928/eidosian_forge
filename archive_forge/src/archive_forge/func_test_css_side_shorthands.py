import pytest
from pandas.errors import CSSWarning
import pandas._testing as tm
from pandas.io.formats.css import CSSResolver
@pytest.mark.parametrize('shorthand,expansions', [('margin', ['margin-top', 'margin-right', 'margin-bottom', 'margin-left']), ('padding', ['padding-top', 'padding-right', 'padding-bottom', 'padding-left']), ('border-width', ['border-top-width', 'border-right-width', 'border-bottom-width', 'border-left-width']), ('border-color', ['border-top-color', 'border-right-color', 'border-bottom-color', 'border-left-color']), ('border-style', ['border-top-style', 'border-right-style', 'border-bottom-style', 'border-left-style'])])
def test_css_side_shorthands(shorthand, expansions):
    top, right, bottom, left = expansions
    assert_resolves(f'{shorthand}: 1pt', {top: '1pt', right: '1pt', bottom: '1pt', left: '1pt'})
    assert_resolves(f'{shorthand}: 1pt 4pt', {top: '1pt', right: '4pt', bottom: '1pt', left: '4pt'})
    assert_resolves(f'{shorthand}: 1pt 4pt 2pt', {top: '1pt', right: '4pt', bottom: '2pt', left: '4pt'})
    assert_resolves(f'{shorthand}: 1pt 4pt 2pt 0pt', {top: '1pt', right: '4pt', bottom: '2pt', left: '0pt'})
    with tm.assert_produces_warning(CSSWarning):
        assert_resolves(f'{shorthand}: 1pt 1pt 1pt 1pt 1pt', {})