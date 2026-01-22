from pathlib import Path
import io
import pytest
from matplotlib import ft2font
from matplotlib.testing.decorators import check_figures_equal
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
def test_fallback_errors():
    file_name = fm.findfont('DejaVu Sans')
    with pytest.raises(TypeError, match='Fallback list must be a list'):
        ft2font.FT2Font(file_name, _fallback_list=(0,))
    with pytest.raises(TypeError, match='Fallback fonts must be FT2Font objects.'):
        ft2font.FT2Font(file_name, _fallback_list=[0])