import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
def test_listproperty_is_none():
    from kivy.properties import ListProperty
    l1 = ListProperty(None)
    l1.set_name(wid, 'l1')
    l1.link_eagerly(wid)
    assert l1.get(wid) is None
    l2 = ListProperty([1, 2, 3], allownone=True)
    l2.set_name(wid, 'l2')
    l2.link_eagerly(wid)
    l2.set(wid, None)
    assert l2.get(wid) is None