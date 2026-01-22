from kivy.clock import Clock
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.properties import BooleanProperty, ListProperty, ObjectProperty, \
Generator to iterate over all nodes from `node` and down whether
        expanded or not. If `node` is `None`, the generator start with
        :attr:`root`.
        