import numpy as np
import pandas as pd
import param
from param import rx
from panel.layout import Row, WidgetBox
from panel.pane.base import PaneBase
from panel.param import ReactiveExpr
from panel.widgets import IntSlider
def test_reactive_widget_input():
    slider = IntSlider()
    expr = ReactiveExpr(rx(slider))
    assert slider in expr.widgets