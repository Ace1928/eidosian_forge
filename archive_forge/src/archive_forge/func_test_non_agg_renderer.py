import warnings
import numpy as np
from numpy.testing import assert_array_equal
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea
from matplotlib.patches import Rectangle
@pytest.mark.backend('pdf')
def test_non_agg_renderer(monkeypatch, recwarn):
    unpatched_init = mpl.backend_bases.RendererBase.__init__

    def __init__(self, *args, **kwargs):
        assert isinstance(self, mpl.backends.backend_pdf.RendererPdf)
        unpatched_init(self, *args, **kwargs)
    monkeypatch.setattr(mpl.backend_bases.RendererBase, '__init__', __init__)
    fig, ax = plt.subplots()
    fig.tight_layout()