from plotly import exceptions
from plotly.graph_objs import graph_objs
from plotly.figure_factory import utils
def make_increasing_ohlc(open, high, low, close, dates, **kwargs):
    """
    Makes increasing ohlc sticks

    _make_increasing_ohlc() and _make_decreasing_ohlc separate the
    increasing trace from the decreasing trace so kwargs (such as
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

    :rtype (trace) ohlc_incr_data: Scatter trace of all increasing ohlc
        sticks.
    """
    flat_increase_x, flat_increase_y, text_increase = _OHLC(open, high, low, close, dates).get_increase()
    if 'name' in kwargs:
        showlegend = True
    else:
        kwargs.setdefault('name', 'Increasing')
        showlegend = False
    kwargs.setdefault('line', dict(color=_DEFAULT_INCREASING_COLOR, width=1))
    kwargs.setdefault('text', text_increase)
    ohlc_incr = dict(type='scatter', x=flat_increase_x, y=flat_increase_y, mode='lines', showlegend=showlegend, **kwargs)
    return ohlc_incr