from kivy.clock import Clock
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.properties import BooleanProperty, ListProperty, ObjectProperty, \
def on_root_options(self, instance, value):
    if not self.root:
        return
    for key, value in value.items():
        setattr(self.root, key, value)