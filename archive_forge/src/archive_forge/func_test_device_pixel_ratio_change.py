import copy
import importlib
import os
import signal
import sys
from datetime import date, datetime
from unittest import mock
import pytest
import matplotlib
from matplotlib import pyplot as plt
from matplotlib._pylab_helpers import Gcf
from matplotlib import _c_internal_utils
@pytest.mark.backend('QtAgg', skip_on_importerror=True)
def test_device_pixel_ratio_change():
    """
    Make sure that if the pixel ratio changes, the figure dpi changes but the
    widget remains the same logical size.
    """
    prop = 'matplotlib.backends.backend_qt.FigureCanvasQT.devicePixelRatioF'
    with mock.patch(prop) as p:
        p.return_value = 3
        fig = plt.figure(figsize=(5, 2), dpi=120)
        qt_canvas = fig.canvas
        qt_canvas.show()

        def set_device_pixel_ratio(ratio):
            p.return_value = ratio
            screen.logicalDotsPerInchChanged.emit(96)
            qt_canvas.draw()
            qt_canvas.flush_events()
            assert qt_canvas.device_pixel_ratio == ratio
        qt_canvas.manager.show()
        size = qt_canvas.size()
        screen = qt_canvas.window().windowHandle().screen()
        set_device_pixel_ratio(3)
        assert fig.dpi == 360
        assert qt_canvas.renderer.width == 1800
        assert qt_canvas.renderer.height == 720
        assert size.width() == 600
        assert size.height() == 240
        assert qt_canvas.get_width_height() == (600, 240)
        assert (fig.get_size_inches() == (5, 2)).all()
        set_device_pixel_ratio(2)
        assert fig.dpi == 240
        assert qt_canvas.renderer.width == 1200
        assert qt_canvas.renderer.height == 480
        assert size.width() == 600
        assert size.height() == 240
        assert qt_canvas.get_width_height() == (600, 240)
        assert (fig.get_size_inches() == (5, 2)).all()
        set_device_pixel_ratio(1.5)
        assert fig.dpi == 180
        assert qt_canvas.renderer.width == 900
        assert qt_canvas.renderer.height == 360
        assert size.width() == 600
        assert size.height() == 240
        assert qt_canvas.get_width_height() == (600, 240)
        assert (fig.get_size_inches() == (5, 2)).all()