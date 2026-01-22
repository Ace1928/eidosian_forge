import base64
import copy
import pickle
import uuid
from collections import namedtuple
from dash.exceptions import PreventUpdate
import holoviews as hv
from holoviews.core.decollate import (
from holoviews.plotting.plotly import DynamicMap, PlotlyRenderer
from holoviews.plotting.plotly.callbacks import (
from holoviews.plotting.plotly.util import clean_internal_figure_properties
from holoviews.streams import Derived, History
import plotly.graph_objects as go
from dash import callback_context
from dash.dependencies import Input, Output, State
def decode_store_data(store_data):
    """
    Decode a dict that was encoded by the encode_store_data function.

    Args:
        store_data: dict that was encoded by encode_store_data

    Returns:
        decoded dict
    """
    return pickle.loads(base64.b64decode(store_data['pickled']))