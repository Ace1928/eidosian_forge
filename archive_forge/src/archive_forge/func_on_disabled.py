from kivy.animation import Animation
from kivy.clock import Clock
from kivy.graphics import CanvasBase, Color, Ellipse, ScissorPush, ScissorPop
from kivy.properties import BooleanProperty, ListProperty, NumericProperty, \
from kivy.uix.relativelayout import RelativeLayout
def on_disabled(self, instance, value):
    if value:
        self.ripple_fade()
    return super(TouchRippleButtonBehavior, self).on_disabled(instance, value)