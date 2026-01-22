import ast
import runpy
from inspect import isclass
from pathlib import Path
import pytest
import panel as pn
def is_panel_pane(attr):
    pane = getattr(pn.pane, attr)
    return isclass(pane) and issubclass(pane, pn.pane.PaneBase)