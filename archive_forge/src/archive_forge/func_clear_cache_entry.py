import json
import traceback
from contextvars import copy_context
from _plotly_utils.utils import PlotlyJSONEncoder
from dash._callback_context import context_value
from dash._utils import AttributeDict
from dash.exceptions import PreventUpdate
from dash.long_callback.managers import BaseLongCallbackManager
def clear_cache_entry(self, key):
    self.handle.backend.delete(key)