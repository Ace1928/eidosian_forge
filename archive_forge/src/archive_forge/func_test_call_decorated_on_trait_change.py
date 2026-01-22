from unittest.mock import patch
import os
from collections import OrderedDict
import pytest
import ipywidgets as widgets
from traitlets import TraitError, Float
from ipywidgets import (interact, interact_manual, interactive,
from .utils import setup, teardown
def test_call_decorated_on_trait_change(clear_display):
    """test calling @interact decorated functions"""
    d = {}
    with patch.object(interaction, 'display', record_display):

        @interact
        def foo(a='default'):
            d['a'] = a
            return a
    assert len(displayed) == 2
    w = displayed[1].children[0]
    check_widget(w, cls=widgets.Text, value='default')
    a = foo('hello')
    assert a == 'hello'
    assert d['a'] == 'hello'
    with patch.object(interaction, 'display', record_display):
        w.value = 'called'
    assert d['a'] == 'called'
    assert len(displayed) == 3
    assert w.value == displayed[-1]