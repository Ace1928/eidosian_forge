import numpy as np
import pandas as pd
import param
from param import rx
from panel.layout import Row, WidgetBox
from panel.pane.base import PaneBase
from panel.param import ReactiveExpr
from panel.widgets import IntSlider
def test_reactive_widget_order():
    slider1 = IntSlider(name='Slider1')
    slider2 = IntSlider(name='Slider2')
    expr = ReactiveExpr(rx(slider1) + rx(slider2))
    assert list(expr.widgets) == [slider1, slider2]