from plotly.figure_factory import utils
from plotly.figure_factory._ohlc import (
from plotly.graph_objs import graph_objs
def make_increasing_candle(open, high, low, close, dates, **kwargs):
    """
    Makes boxplot trace for increasing candlesticks

    _make_increasing_candle() and _make_decreasing_candle separate the
    increasing traces from the decreasing traces so kwargs (such as
    color) can be passed separately to increasing or decreasing traces
    when direction is set to 'increasing' or 'decreasing' in
    FigureFactory.create_candlestick()

    :param (list) open: opening values
    :param (list) high: high values
    :param (list) low: low values
    :param (list) close: closing values
    :param (list) dates: list of datetime objects. Default: None
    :param kwargs: kwargs to be passed to increasing trace via
        plotly.graph_objs.Scatter.

    :rtype (list) candle_incr_data: list of the box trace for
        increasing candlesticks.
    """
    increase_x, increase_y = _Candlestick(open, high, low, close, dates, **kwargs).get_candle_increase()
    if 'line' in kwargs:
        kwargs.setdefault('fillcolor', kwargs['line']['color'])
    else:
        kwargs.setdefault('fillcolor', _DEFAULT_INCREASING_COLOR)
    if 'name' in kwargs:
        kwargs.setdefault('showlegend', True)
    else:
        kwargs.setdefault('showlegend', False)
    kwargs.setdefault('name', 'Increasing')
    kwargs.setdefault('line', dict(color=_DEFAULT_INCREASING_COLOR))
    candle_incr_data = dict(type='box', x=increase_x, y=increase_y, whiskerwidth=0, boxpoints=False, **kwargs)
    return [candle_incr_data]