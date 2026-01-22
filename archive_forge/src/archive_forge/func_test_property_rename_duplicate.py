import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
def test_property_rename_duplicate():
    from kivy.event import EventDispatcher
    from kivy.properties import ObjectProperty

    class Event(EventDispatcher):
        b = ObjectProperty(5)
        a = b
    event = Event()
    counter = 0
    counter2 = 0

    def callback(*args):
        nonlocal counter
        counter += 1

    def callback2(*args):
        nonlocal counter2
        counter2 += 1
    event.fbind('a', callback)
    event.fbind('b', callback2)
    event.a = 12
    assert counter == 1
    assert counter2 == 1
    assert event.a == 12
    assert event.b == 12
    event.b = 14
    assert counter == 2
    assert counter2 == 2
    assert event.a == 14
    assert event.b == 14