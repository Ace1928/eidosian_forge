from numbers import Number
from plotly import exceptions, optional_imports
import plotly.colors as clrs
from plotly.graph_objs import graph_objs
from plotly.subplots import make_subplots
def calc_stats(data):
    """
    Calculate statistics for use in violin plot.
    """
    x = np.asarray(data, float)
    vals_min = np.min(x)
    vals_max = np.max(x)
    q2 = np.percentile(x, 50, interpolation='linear')
    q1 = np.percentile(x, 25, interpolation='lower')
    q3 = np.percentile(x, 75, interpolation='higher')
    iqr = q3 - q1
    whisker_dist = 1.5 * iqr
    d1 = np.min(x[x >= q1 - whisker_dist])
    d2 = np.max(x[x <= q3 + whisker_dist])
    return {'min': vals_min, 'max': vals_max, 'q1': q1, 'q2': q2, 'q3': q3, 'd1': d1, 'd2': d2}