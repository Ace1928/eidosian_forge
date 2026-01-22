from traitlets import (Bool, Int, Float, Unicode, Dict,
from traittypes import Array
from ipywidgets import Widget, Color, widget_serialization, register
from .scales import Scale
from .traits import Date, array_serialization, _array_equal
from .marks import Lines
from ._version import __frontend_version__
import numpy as np
Lasso selector interaction.

    This 2-D selector enables the user to select multiple sets of data points
    by drawing lassos on the figure. A mouse-down starts drawing the lasso and
    after the mouse-up the lasso is closed and the `selected` attribute of each
    mark gets updated with the data in the lasso.

    The user can select (de-select) by clicking on lassos and can delete them
    (and their associated data) by pressing the 'Delete' button.

    Attributes
    ----------
    marks: List of marks which are instances of {Lines, Scatter} (default: [])
        List of marks on which lasso selector will be applied.
    color: Color (default: None)
        Color of the lasso.
    