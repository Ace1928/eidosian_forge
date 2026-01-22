from kivy.compat import iteritems
from kivy.logger import Logger
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import (StringProperty, ObjectProperty, AliasProperty,
from kivy.animation import Animation, AnimationTransition
from kivy.uix.relativelayout import RelativeLayout
from kivy.lang import Builder
from kivy.graphics import (RenderContext, Rectangle, Fbo,
def on_complete(self):
    self.screen_in.pos = self.manager.pos
    self.screen_out.pos = self.manager.pos
    for screen in (self.screen_in, self.screen_out):
        for canvas in (screen.canvas.before, screen.canvas.after):
            canvas.remove_group('swaptransition_scale')
    super(SwapTransition, self).on_complete()