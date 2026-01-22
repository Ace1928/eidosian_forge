from datetime import date, datetime
from pathlib import Path
import numpy as np
import pytest
from bokeh.models.widgets import FileInput as BkFileInput
from panel import config
from panel.widgets import (
def test_text_input(document, comm):
    text = TextInput(value='ABC', name='Text:')
    widget = text.get_root(document, comm=comm)
    assert isinstance(widget, text._widget_type)
    assert widget.value == 'ABC'
    assert widget.title == 'Text:'
    text._process_events({'value': 'CBA'})
    assert text.value == 'CBA'
    text.value = 'A'
    assert widget.value == 'A'