import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
@pytest.mark.parametrize('set_name', [True, False])
def test_bounded_numeric_property_error_value(self, set_name):
    from kivy.properties import BoundedNumericProperty
    bnp = BoundedNumericProperty(0, min=-5, max=5, errorvalue=1)
    if set_name:
        bnp.set_name(wid, 'bnp')
        bnp.link_eagerly(wid)
    else:
        bnp.link(wid, 'bnp')
        bnp.link_deps(wid, 'bnp')
    bnp.set(wid, 1)
    self.assertEqual(bnp.get(wid), 1)
    bnp.set(wid, 5)
    self.assertEqual(bnp.get(wid), 5)
    bnp.set(wid, 6)
    self.assertEqual(bnp.get(wid), 1)
    bnp.set(wid, -5)
    self.assertEqual(bnp.get(wid), -5)
    bnp.set(wid, -6)
    self.assertEqual(bnp.get(wid), 1)