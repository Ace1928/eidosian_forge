from kivy.factory import Factory
from kivy.uix.button import Button
from kivy.properties import (OptionProperty, NumericProperty, ObjectProperty,
from kivy.uix.boxlayout import BoxLayout
def on_sizable_from(self, instance, sizable_from):
    if not instance._container:
        return
    sup = super(Splitter, instance)
    _strp = instance._strip
    if _strp:
        _strp.unbind(on_touch_down=instance.strip_down)
        _strp.unbind(on_touch_move=instance.strip_move)
        _strp.unbind(on_touch_up=instance.strip_up)
        self.unbind(disabled=_strp.setter('disabled'))
        sup.remove_widget(instance._strip)
    cls = instance.strip_cls
    if not isinstance(_strp, cls):
        if isinstance(cls, str):
            cls = Factory.get(cls)
        instance._strip = _strp = cls()
    sz_frm = instance.sizable_from[0]
    if sz_frm in ('l', 'r'):
        _strp.size_hint = (None, 1)
        _strp.width = instance.strip_size
        instance.orientation = 'horizontal'
        instance.unbind(strip_size=_strp.setter('width'))
        instance.bind(strip_size=_strp.setter('width'))
    else:
        _strp.size_hint = (1, None)
        _strp.height = instance.strip_size
        instance.orientation = 'vertical'
        instance.unbind(strip_size=_strp.setter('height'))
        instance.bind(strip_size=_strp.setter('height'))
    index = 1
    if sz_frm in ('r', 'b'):
        index = 0
    sup.add_widget(_strp, index)
    _strp.bind(on_touch_down=instance.strip_down)
    _strp.bind(on_touch_move=instance.strip_move)
    _strp.bind(on_touch_up=instance.strip_up)
    _strp.disabled = self.disabled
    self.bind(disabled=_strp.setter('disabled'))