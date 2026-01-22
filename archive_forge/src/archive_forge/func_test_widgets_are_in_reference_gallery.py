import ast
import runpy
from inspect import isclass
from pathlib import Path
import pytest
import panel as pn
@ref_available
def test_widgets_are_in_reference_gallery():
    exceptions = {'Ace', 'CompositeWidget', 'Widget', 'ToggleGroup', 'NumberInput', 'Spinner'}
    docs = {f.with_suffix('').name for g in ('indicators', 'widgets') for f in (REF_PATH / g).iterdir()}

    def is_panel_widget(attr):
        widget = getattr(pn.widgets, attr)
        return isclass(widget) and issubclass(widget, pn.widgets.Widget)
    widgets = set(filter(is_panel_widget, dir(pn.widgets)))
    assert widgets - exceptions - docs == set()