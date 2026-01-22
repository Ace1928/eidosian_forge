import math
import warnings
import matplotlib.dates
def get_markerstyle_from_collection(props):
    markerstyle = dict(alpha=None, facecolor=convert_rgba_array(props['styles']['facecolor']), marker=convert_path_array(props['paths']), edgewidth=convert_linewidth_array(props['styles']['linewidth']), markersize=convert_size_array(props['mplobj'].get_sizes()), edgecolor=convert_rgba_array(props['styles']['edgecolor']))
    return markerstyle