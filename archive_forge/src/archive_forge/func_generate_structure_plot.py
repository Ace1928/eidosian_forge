from __future__ import annotations
import logging # isort:skip
from itertools import permutations
from typing import TYPE_CHECKING
from bokeh.core.properties import UnsetValueError
from bokeh.layouts import column
from bokeh.models import (
def generate_structure_plot(f: Model) -> Model:
    """  Given a bokeh model f, return a model that displays the graph of its submodels.

    Clicking on the nodes of the graph reveals the attributes of that submodel.

    """
    return _BokehStructureGraph(f).model