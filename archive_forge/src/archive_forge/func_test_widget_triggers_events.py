import param
import pytest
from panel.io import block_comm
from panel.layout import Row
from panel.links import CallbackGenerator
from panel.tests.util import check_layoutable_properties
from panel.util import param_watchers
from panel.widgets import (
from panel.widgets.tables import BaseTable
def test_widget_triggers_events(document, comm):
    """
    Ensure widget events don't get swallowed in comm mode
    """
    text = TextInput(value='ABC', name='Text:')
    widget = text.get_root(document, comm=comm)
    document.add_root(widget)
    document.hold()
    document.callbacks._held_events = document.callbacks._held_events[:-1]
    with block_comm():
        text.value = '123'
    assert len(document.callbacks._held_events) == 1
    event = document.callbacks._held_events[0]
    assert event.attr == 'value'
    assert event.model is widget
    assert event.new == '123'