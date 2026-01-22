from unittest import TestCase
from traitlets import TraitError
from ipywidgets.widgets import Accordion, Tab, Stack, HTML
def test_selected_index_default(self):
    widget = self.widget(self.children)
    state = widget.get_state()
    assert state['selected_index'] is None