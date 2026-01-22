import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
def test_property_rebind(self):
    from kivy.uix.label import Label
    from kivy.uix.togglebutton import ToggleButton
    from kivy.lang import Builder
    from kivy.properties import ObjectProperty, DictProperty, AliasProperty
    from kivy.clock import Clock

    class ObjWidget(Label):
        button = ObjectProperty(None, rebind=True, allownone=True)

    class ObjWidgetRebindFalse(Label):
        button = ObjectProperty(None, rebind=False, allownone=True)

    class DictWidget(Label):
        button = DictProperty({'button': None}, rebind=True, allownone=True)

    class DictWidgetFalse(Label):
        button = DictProperty({'button': None}, rebind=False)

    class AliasWidget(Label):
        _button = None

        def setter(self, value):
            self._button = value
            return True

        def getter(self):
            return self._button
        button = AliasProperty(getter, setter, rebind=True)
    Builder.load_string("\n<ObjWidget>:\n    text: self.button.state if self.button is not None else 'Unset'\n\n<ObjWidgetRebindFalse>:\n    text: self.button.state if self.button is not None else 'Unset'\n\n<AliasWidget>:\n    text: self.button.state if self.button is not None else 'Unset'\n\n<DictWidget>:\n    text: self.button.button.state if self.button.button is not None else 'Unset'\n\n<DictWidgetFalse>:\n    text: self.button.button.state if self.button.button is not None else 'Unset'\n")
    obj = ObjWidget()
    obj_false = ObjWidgetRebindFalse()
    dict_rebind = DictWidget()
    dict_false = DictWidgetFalse()
    alias_rebind = AliasWidget()
    button = ToggleButton()
    Clock.tick()
    self.assertEqual(obj.text, 'Unset')
    self.assertEqual(obj_false.text, 'Unset')
    self.assertEqual(dict_rebind.text, 'Unset')
    self.assertEqual(dict_false.text, 'Unset')
    self.assertEqual(alias_rebind.text, 'Unset')
    obj.button = button
    obj_false.button = button
    dict_rebind.button.button = button
    dict_false.button.button = button
    alias_rebind.button = button
    Clock.tick()
    self.assertEqual(obj.text, 'normal')
    self.assertEqual(obj_false.text, 'normal')
    self.assertEqual(dict_rebind.text, 'normal')
    self.assertEqual(dict_false.text, 'Unset')
    self.assertEqual(alias_rebind.text, 'normal')
    button.state = 'down'
    Clock.tick()
    self.assertEqual(obj.text, 'down')
    self.assertEqual(obj_false.text, 'normal')
    self.assertEqual(dict_rebind.text, 'down')
    self.assertEqual(dict_false.text, 'Unset')
    self.assertEqual(alias_rebind.text, 'down')
    button.state = 'normal'
    Clock.tick()
    self.assertEqual(obj.text, 'normal')
    self.assertEqual(obj_false.text, 'normal')
    self.assertEqual(dict_rebind.text, 'normal')
    self.assertEqual(dict_false.text, 'Unset')
    self.assertEqual(alias_rebind.text, 'normal')
    obj.button = None
    obj_false.button = None
    dict_rebind.button.button = None
    dict_false.button.button = None
    alias_rebind.button = None
    Clock.tick()
    self.assertEqual(obj.text, 'Unset')
    self.assertEqual(obj_false.text, 'Unset')
    self.assertEqual(dict_rebind.text, 'Unset')
    self.assertEqual(dict_false.text, 'Unset')
    self.assertEqual(alias_rebind.text, 'Unset')