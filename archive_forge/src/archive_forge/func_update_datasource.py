import random
from panel import (
from panel.pane import HTML
from panel.widgets import IntSlider, Trend
def update_datasource():
    new_x = max(data['x']) + 1
    old_y = data['y'][-1]
    new_y = random.uniform(-old_y * 0.05, old_y * 0.05) + old_y * 1.01
    trend.stream({'x': [new_x], 'y': [new_y]}, rollover=50)
    y_series = data['y']
    trend.value = y_series[-1]
    change = y_series[-1] / y_series[-2] - 1
    trend.value_change = change