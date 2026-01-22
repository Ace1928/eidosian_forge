from ast import literal_eval
import json
from ipywidgets import register, CallbackDispatcher, DOMWidget
from traitlets import Any, Bool, Int, Unicode
from ..data_utils.binary_transfer import data_buffer_serialization
from ._frontend import module_name, module_version
from .debounce import debounce
def store_selection(widget_instance, payload):
    """Callback for storing data on click"""
    try:
        if payload.get('data') and payload['data'].get('object'):
            datum = payload['data']['object']
            widget_instance.selected_data.append(datum)
        else:
            widget_instance.selected_data = []
    except Exception as e:
        widget_instance.handler_exception = e