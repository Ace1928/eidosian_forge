import numpy as np
import pytest
from holoviews import (
from holoviews.element import Curve, HLine, Image
from holoviews.element.comparison import ComparisonTestCase
@pytest.mark.usefixtures('mpl_backend')
def test_histogram_image_hline_overlay(self):
    image = Image(np.arange(100).reshape(10, 10))
    overlay = image * HLine(y=0)
    element = overlay.hist()
    assert isinstance(element, AdjointLayout)
    assert element.main == overlay