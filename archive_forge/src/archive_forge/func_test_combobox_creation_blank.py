import inspect
import pytest
from ..widget_string import Combobox, Text
def test_combobox_creation_blank():
    w = Combobox()
    assert w.value == ''
    assert w.options == ()
    assert w.ensure_option == False