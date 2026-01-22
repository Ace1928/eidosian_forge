from traitlets import (Bool, Int, Float, Unicode, Dict,
from traittypes import Array
from ipywidgets import Widget, Color, widget_serialization, register
from .scales import Scale
from .traits import Date, array_serialization, _array_equal
from .marks import Lines
from ._version import __frontend_version__
import numpy as np
def register_interaction(key=None):
    """Decorator registering an interaction class in the registry.

    If no key is provided, the class name is used as a key. A key is provided
    for each core bqplot interaction type so that the frontend can use this
    key regardless of the kernel language.
    """

    def wrap(interaction):
        name = key if key is not None else interaction.__module__ + interaction.__name__
        interaction.types[name] = interaction
        return interaction
    return wrap