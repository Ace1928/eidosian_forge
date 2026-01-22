from plotly import exceptions
from plotly.graph_objs import graph_objs
from plotly.figure_factory import utils
def make_decreasing_ohlc(open, high, low, close, dates, **kwargs):
    """
    Makes decreasing ohlc sticks

    :param (list) open: opening values
    :param (list) high: high values
    :param (list) low: low values
    :param (list) close: closing values
    :param (list) dates: list of datetime objects. Default: None
    :param kwargs: kwargs to be passed to increasing trace via
        plotly.graph_objs.Scatter.

    :rtype (trace) ohlc_decr_data: Scatter trace of all decreasing ohlc
        sticks.
    """
    flat_decrease_x, flat_decrease_y, text_decrease = _OHLC(open, high, low, close, dates).get_decrease()
    kwargs.setdefault('line', dict(color=_DEFAULT_DECREASING_COLOR, width=1))
    kwargs.setdefault('text', text_decrease)
    kwargs.setdefault('showlegend', False)
    kwargs.setdefault('name', 'Decreasing')
    ohlc_decr = dict(type='scatter', x=flat_decrease_x, y=flat_decrease_y, mode='lines', **kwargs)
    return ohlc_decr