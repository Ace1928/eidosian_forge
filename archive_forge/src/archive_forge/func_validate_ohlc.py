from plotly import exceptions
from plotly.graph_objs import graph_objs
from plotly.figure_factory import utils
def validate_ohlc(open, high, low, close, direction, **kwargs):
    """
    ohlc and candlestick specific validations

    Specifically, this checks that the high value is the greatest value and
    the low value is the lowest value in each unit.

    See FigureFactory.create_ohlc() or FigureFactory.create_candlestick()
    for params

    :raises: (PlotlyError) If the high value is not the greatest value in
        each unit.
    :raises: (PlotlyError) If the low value is not the lowest value in each
        unit.
    :raises: (PlotlyError) If direction is not 'increasing' or 'decreasing'
    """
    for lst in [open, low, close]:
        for index in range(len(high)):
            if high[index] < lst[index]:
                raise exceptions.PlotlyError('Oops! Looks like some of your high values are less the corresponding open, low, or close values. Double check that your data is entered in O-H-L-C order')
    for lst in [open, high, close]:
        for index in range(len(low)):
            if low[index] > lst[index]:
                raise exceptions.PlotlyError('Oops! Looks like some of your low values are greater than the corresponding high, open, or close values. Double check that your data is entered in O-H-L-C order')
    direction_opts = ('increasing', 'decreasing', 'both')
    if direction not in direction_opts:
        raise exceptions.PlotlyError("direction must be defined as 'increasing', 'decreasing', or 'both'")