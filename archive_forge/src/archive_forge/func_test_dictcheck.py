import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
@pytest.mark.parametrize('set_name', [True, False])
def test_dictcheck(self, set_name):
    from kivy.properties import DictProperty
    a = DictProperty()
    if set_name:
        a.set_name(wid, 'a')
        a.link_eagerly(wid)
    else:
        a.link(wid, 'a')
        a.link_deps(wid, 'a')
    self.assertEqual(a.get(wid), {})
    a.set(wid, {'foo': 'bar'})
    self.assertEqual(a.get(wid), {'foo': 'bar'})