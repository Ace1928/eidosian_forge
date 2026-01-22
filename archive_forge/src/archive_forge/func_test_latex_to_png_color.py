from contextlib import contextmanager
from unittest.mock import patch
import pytest
from IPython.lib import latextools
from IPython.testing.decorators import (
from IPython.utils.process import FindCmdError
@skipif_not_matplotlib
@onlyif_cmds_exist('latex', 'dvipng')
def test_latex_to_png_color():
    """
    Test color settings for latex_to_png.
    """
    latex_string = '$x^2$'
    default_value = latextools.latex_to_png(latex_string, wrap=False)
    default_hexblack = latextools.latex_to_png(latex_string, wrap=False, color='#000000')
    dvipng_default = latextools.latex_to_png_dvipng(latex_string, False)
    dvipng_black = latextools.latex_to_png_dvipng(latex_string, False, 'Black')
    assert dvipng_default == dvipng_black
    mpl_default = latextools.latex_to_png_mpl(latex_string, False)
    mpl_black = latextools.latex_to_png_mpl(latex_string, False, 'Black')
    assert mpl_default == mpl_black
    assert default_value in [dvipng_black, mpl_black]
    assert default_hexblack in [dvipng_black, mpl_black]
    dvipng_maroon = latextools.latex_to_png_dvipng(latex_string, False, 'Maroon')
    assert dvipng_black != dvipng_maroon
    mpl_maroon = latextools.latex_to_png_mpl(latex_string, False, 'Maroon')
    assert mpl_black != mpl_maroon
    mpl_white = latextools.latex_to_png_mpl(latex_string, False, 'White')
    mpl_hexwhite = latextools.latex_to_png_mpl(latex_string, False, '#FFFFFF')
    assert mpl_white == mpl_hexwhite
    mpl_white_scale = latextools.latex_to_png_mpl(latex_string, False, 'White', 1.2)
    assert mpl_white != mpl_white_scale