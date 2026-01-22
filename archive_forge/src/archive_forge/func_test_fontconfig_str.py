import pytest
from matplotlib.font_manager import FontProperties
def test_fontconfig_str():
    """Test FontProperties string conversions for correctness."""
    test = 'defaults '
    s = 'sans\\-serif:style=normal:variant=normal:weight=normal:stretch=normal:size=12.0'
    font = FontProperties(s)
    right = FontProperties()
    for k in keys:
        assert getattr(font, k)() == getattr(right, k)(), test + k
    test = 'full '
    s = 'serif-24:style=oblique:variant=small-caps:weight=bold:stretch=expanded'
    font = FontProperties(s)
    right = FontProperties(family='serif', size=24, weight='bold', style='oblique', variant='small-caps', stretch='expanded')
    for k in keys:
        assert getattr(font, k)() == getattr(right, k)(), test + k