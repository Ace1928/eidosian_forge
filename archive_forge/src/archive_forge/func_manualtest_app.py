import random
from panel import (
from panel.pane import HTML
from panel.widgets import IntSlider, Trend
def manualtest_app():
    data = {'x': [1, 2, 3, 4, 5], 'y': [3800, 3700, 3800, 3900, 4000]}
    trend = Trend(name='Panel Users', value=4000, value_change=0.51, data=data, height=200, width=200)

    def update_datasource():
        new_x = max(data['x']) + 1
        old_y = data['y'][-1]
        new_y = random.uniform(-old_y * 0.05, old_y * 0.05) + old_y * 1.01
        trend.stream({'x': [new_x], 'y': [new_y]}, rollover=50)
        y_series = data['y']
        trend.value = y_series[-1]
        change = y_series[-1] / y_series[-2] - 1
        trend.value_change = change
    settings_panel = Param(trend, parameters=['height', 'width', 'sizing_mode', 'layout', 'title', 'plot_color', 'plot_type', 'value_change_pos_color', 'value_change_neg_color'], widgets={'height': {'widget_type': IntSlider, 'start': 0, 'end': 800, 'step': 1}, 'width': {'widget_type': IntSlider, 'start': 0, 'end': 800, 'step': 1}}, sizing_mode='fixed', width=400)
    app = Column(HTML('<h1>Panel - Streaming to TrendIndicator<h1>', sizing_mode='stretch_width', styles={'color': 'white', 'padding': '15px', 'background': 'black'}), Row(WidgetBox(settings_panel), trend, sizing_mode='stretch_both'), sizing_mode='stretch_both')
    state.add_periodic_callback(update_datasource, period=50)
    return app