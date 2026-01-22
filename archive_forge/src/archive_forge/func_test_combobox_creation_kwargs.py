import inspect
import pytest
from ..widget_string import Combobox, Text
def test_combobox_creation_kwargs():
    w = Combobox(value='Chocolate', options=['Chocolate', 'Coconut', 'Mint', 'Strawberry', 'Vanilla'], ensure_option=True)
    assert w.value == 'Chocolate'
    assert w.options == ('Chocolate', 'Coconut', 'Mint', 'Strawberry', 'Vanilla')
    assert w.ensure_option == True