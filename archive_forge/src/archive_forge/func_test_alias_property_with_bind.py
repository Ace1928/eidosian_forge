import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
@pytest.mark.parametrize('watch_before_use', [True, False])
def test_alias_property_with_bind(self, watch_before_use):
    from kivy.properties import NumericProperty, AliasProperty

    class CustomAlias(EventDispatcher):
        x = NumericProperty(0)
        width = NumericProperty(100)

        def get_right(self):
            return self.x + self.width

        def set_right(self, value):
            self.x = value - self.width
        right = AliasProperty(get_right, set_right, bind=('x', 'width'), watch_before_use=watch_before_use)

        def __init__(self, **kwargs):
            super(CustomAlias, self).__init__(**kwargs)
            self.callback_called = 0
    wid = CustomAlias()
    self.assertEqual(wid.right, 100)
    wid.x = 500
    self.assertEqual(wid.right, 600)
    wid.width = 50
    self.assertEqual(wid.right, 550)
    wid.right = 100
    self.assertEqual(wid.width, 50)
    self.assertEqual(wid.x, 50)

    def callback(widget, value):
        widget.callback_called += 1
    wid.bind(right=callback)
    wid.x = 100
    self.assertEqual(wid.callback_called, 1)
    wid.x = 100
    self.assertEqual(wid.callback_called, 1)
    wid.width = 900
    self.assertEqual(wid.callback_called, 2)
    wid.right = 700
    self.assertEqual(wid.callback_called, 3)
    wid.right = 700
    self.assertEqual(wid.callback_called, 3)