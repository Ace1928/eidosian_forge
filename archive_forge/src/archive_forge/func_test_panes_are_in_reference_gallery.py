import ast
import runpy
from inspect import isclass
from pathlib import Path
import pytest
import panel as pn
@ref_available
def test_panes_are_in_reference_gallery():
    exceptions = {'PaneBase', 'YT', 'RGGPlot', 'Interactive', 'ICO', 'Image', 'IPyLeaflet', 'ParamFunction', 'ParamMethod', 'ParamRef'}
    docs = {f.with_suffix('').name for f in (REF_PATH / 'panes').iterdir()}

    def is_panel_pane(attr):
        pane = getattr(pn.pane, attr)
        return isclass(pane) and issubclass(pane, pn.pane.PaneBase)
    panes = set(filter(is_panel_pane, dir(pn.pane)))
    assert panes - exceptions - docs == set()