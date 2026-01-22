import numpy as np
import pandas as pd
import param
from param import rx
from panel.layout import Row, WidgetBox
from panel.pane.base import PaneBase
from panel.param import ReactiveExpr
from panel.widgets import IntSlider
def test_reactive_dataframe_attribute_chain(dataframe):
    array = rx(dataframe).str.values.rx.value
    np.testing.assert_array_equal(array, dataframe.str.values)