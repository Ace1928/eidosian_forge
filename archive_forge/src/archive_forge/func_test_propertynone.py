import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
@pytest.mark.parametrize('set_name', [True, False])
def test_propertynone(self, set_name):
    from kivy.properties import NumericProperty
    a = NumericProperty(0, allownone=True)
    if set_name:
        a.set_name(wid, 'a')
        a.link_eagerly(wid)
    else:
        a.link(wid, 'a')
        a.link_deps(wid, 'a')
    self.assertEqual(a.get(wid), 0)
    try:
        a.set(wid, None)
        self.assertEqual(a.get(wid), None)
    except ValueError:
        pass
    a.set(wid, 1)
    self.assertEqual(a.get(wid), 1)