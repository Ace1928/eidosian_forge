from plotly.figure_factory import utils
from plotly.figure_factory._ohlc import (
from plotly.graph_objs import graph_objs
def make_decreasing_candle(open, high, low, close, dates, **kwargs):
    """
    Makes boxplot trace for decreasing candlesticks

    :param (list) open: opening values
    :param (list) high: high values
    :param (list) low: low values
    :param (list) close: closing values
    :param (list) dates: list of datetime objects. Default: None
    :param kwargs: kwargs to be passed to decreasing trace via
        plotly.graph_objs.Scatter.

    :rtype (list) candle_decr_data: list of the box trace for
        decreasing candlesticks.
    """
    decrease_x, decrease_y = _Candlestick(open, high, low, close, dates, **kwargs).get_candle_decrease()
    if 'line' in kwargs:
        kwargs.setdefault('fillcolor', kwargs['line']['color'])
    else:
        kwargs.setdefault('fillcolor', _DEFAULT_DECREASING_COLOR)
    kwargs.setdefault('showlegend', False)
    kwargs.setdefault('line', dict(color=_DEFAULT_DECREASING_COLOR))
    kwargs.setdefault('name', 'Decreasing')
    candle_decr_data = dict(type='box', x=decrease_x, y=decrease_y, whiskerwidth=0, boxpoints=False, **kwargs)
    return [candle_decr_data]