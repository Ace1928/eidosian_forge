import ast
import runpy
from inspect import isclass
from pathlib import Path
import pytest
import panel as pn
@ref_available
def test_layouts_are_in_reference_gallery():
    exceptions = {'ListPanel', 'Panel'}
    docs = {f.with_suffix('').name for f in (REF_PATH / 'layouts').iterdir()}

    def is_panel_layout(attr):
        layout = getattr(pn.layout, attr)
        return isclass(layout) and issubclass(layout, pn.layout.Panel)
    layouts = set(filter(is_panel_layout, dir(pn.layout)))
    assert layouts - exceptions - docs == set()