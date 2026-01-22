from kivy.core.text import DEFAULT_FONT
from kivy.uix.modalview import ModalView
from kivy.properties import (StringProperty, ObjectProperty, OptionProperty,
def on__container(self, instance, value):
    if value is None or self.content is None:
        return
    self._container.clear_widgets()
    self._container.add_widget(self.content)