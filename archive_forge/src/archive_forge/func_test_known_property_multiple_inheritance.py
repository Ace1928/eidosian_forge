import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
def test_known_property_multiple_inheritance():

    class Behavior:

        def __init__(self, name):
            print(f'Behavior: {self}, name={name}')
            super().__init__()

    class Widget2(Behavior, EventDispatcher):
        pass

    class Widget3(EventDispatcher, Behavior):
        pass
    with pytest.raises(TypeError) as cm:
        EventDispatcher(name='Pasta')
    assert "Properties ['name'] passed to __init__ may not be existing" in str(cm.value)
    Widget2(name='Pasta')
    Widget3(name='Pasta')