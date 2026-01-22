from unittest.mock import patch
import os
from collections import OrderedDict
import pytest
import ipywidgets as widgets
from traitlets import TraitError, Float
from ipywidgets import (interact, interact_manual, interactive,
from .utils import setup, teardown
def test_call_interact_on_trait_changed_none_return(clear_display):

    def foo(a='default'):
        pass
    with patch.object(interaction, 'display', record_display):
        ifoo = interact(foo)
    assert len(displayed) == 1
    w = displayed[0].children[0]
    check_widget(w, cls=widgets.Text, value='default')
    with patch.object(interaction, 'display', record_display):
        w.value = 'called'
    assert len(displayed) == 1