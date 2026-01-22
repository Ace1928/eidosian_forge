import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
@pytest.mark.parametrize('set_name', [True, False])
def test_listcheck(self, set_name):
    from kivy.properties import ListProperty
    a = ListProperty()
    if set_name:
        a.set_name(wid, 'a')
        a.link_eagerly(wid)
    else:
        a.link(wid, 'a')
        a.link_deps(wid, 'a')
    self.assertEqual(a.get(wid), [])
    a.set(wid, [1, 2, 3])
    self.assertEqual(a.get(wid), [1, 2, 3])