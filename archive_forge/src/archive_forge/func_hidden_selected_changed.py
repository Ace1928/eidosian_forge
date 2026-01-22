from traitlets import (Bool, Int, Float, Unicode, Dict,
from traittypes import Array
from ipywidgets import Widget, Color, widget_serialization, register
from .scales import Scale
from .traits import Date, array_serialization, _array_equal
from .marks import Lines
from ._version import __frontend_version__
import numpy as np
def hidden_selected_changed(self, name, selected):
    actual_selected = {}
    if self.read_json is None:
        self.selected = self._selected
    else:
        for key in self._selected:
            actual_selected[key] = [self.read_json(elem) for elem in self._selected[key]]
        self.selected = actual_selected