from ast import literal_eval
import json
from ipywidgets import register, CallbackDispatcher, DOMWidget
from traitlets import Any, Bool, Int, Unicode
from ..data_utils.binary_transfer import data_buffer_serialization
from ._frontend import module_name, module_version
from .debounce import debounce
def on_drag_start(self, callback, remove=False):
    self._drag_start_handlers.register_callback(callback, remove=remove)