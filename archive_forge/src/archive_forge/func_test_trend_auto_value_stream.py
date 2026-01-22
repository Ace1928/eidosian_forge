import random
from panel import (
from panel.pane import HTML
from panel.widgets import IntSlider, Trend
def test_trend_auto_value_stream(document, comm):
    data = {'x': [1, 2, 3, 4, 5], 'y': [3800, 3700, 3800, 3900, 4000]}
    trend = Trend(data=data)
    model = trend.get_root(document, comm)
    trend.stream({'x': [6], 'y': [4100]}, rollover=5)
    assert model.value == 4100
    assert model.value_change == 4100 / 4000 - 1
    assert len(model.source.data['x']) == 5
    assert model.source.data['x'][-1] == 6