import json
import warnings
import os
from plotly import exceptions, optional_imports
from plotly.files import PLOTLY_DIR
def get_graph_obj(obj, obj_type=None):
    """Returns a new graph object.

    OLD FUNCTION: this will *silently* strip out invalid pieces of the object.
    NEW FUNCTION: no striping of invalid pieces anymore - only raises error
        on unrecognized graph_objs
    """
    from plotly.graph_objs import graph_objs
    try:
        cls = getattr(graph_objs, obj_type)
    except (AttributeError, KeyError):
        raise exceptions.PlotlyError("'{}' is not a recognized graph_obj.".format(obj_type))
    return cls(obj)