import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
@pytest.mark.parametrize('set_name', [True, False])
def test_numeric_string_without_units(self, set_name):
    from kivy.properties import NumericProperty
    a = NumericProperty()
    if set_name:
        a.set_name(wid, 'a')
        a.link_eagerly(wid)
    else:
        a.link(wid, 'a')
        a.link_deps(wid, 'a')
    self.assertEqual(a.get(wid), 0)
    a.set(wid, '2')
    self.assertEqual(a.get(wid), 2)