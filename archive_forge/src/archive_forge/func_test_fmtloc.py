import copy
import matplotlib.pyplot as plt
from matplotlib.scale import (
import matplotlib.scale as mscale
from matplotlib.ticker import AsinhLocator, LogFormatterSciNotation
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import numpy as np
from numpy.testing import assert_allclose
import io
import pytest
def test_fmtloc(self):

    class DummyAxis:

        def __init__(self):
            self.fields = {}

        def set(self, **kwargs):
            self.fields.update(**kwargs)

        def set_major_formatter(self, f):
            self.fields['major_formatter'] = f
    ax0 = DummyAxis()
    s0 = AsinhScale(axis=ax0, base=0)
    s0.set_default_locators_and_formatters(ax0)
    assert isinstance(ax0.fields['major_locator'], AsinhLocator)
    assert isinstance(ax0.fields['major_formatter'], str)
    ax5 = DummyAxis()
    s7 = AsinhScale(axis=ax5, base=5)
    s7.set_default_locators_and_formatters(ax5)
    assert isinstance(ax5.fields['major_locator'], AsinhLocator)
    assert isinstance(ax5.fields['major_formatter'], LogFormatterSciNotation)