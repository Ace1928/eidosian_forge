import pytest
from pandas import Series
from pandas.plotting._matplotlib.style import get_standard_colors
@pytest.mark.parametrize('num_colors, expected', [(1, ['b']), (3, ['b', 'g', 'r']), (4, ['b', 'g', 'r', 'y']), (5, ['b', 'g', 'r', 'y', 'b']), (7, ['b', 'g', 'r', 'y', 'b', 'g', 'r'])])
def test_default_colors_named_from_prop_cycle_string(self, num_colors, expected):
    import matplotlib as mpl
    from matplotlib.pyplot import cycler
    mpl_params = {'axes.prop_cycle': cycler(color='bgry')}
    with mpl.rc_context(rc=mpl_params):
        result = get_standard_colors(num_colors=num_colors)
        assert result == expected