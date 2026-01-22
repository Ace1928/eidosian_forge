from kivy.clock import Clock
from kivy.uix.relativelayout import RelativeLayout
from kivy.properties import (StringProperty, ObjectProperty, ListProperty,
from kivy.graphics import (RenderContext, Fbo, Color, Rectangle,
from kivy.event import EventDispatcher
from kivy.base import EventLoop
from kivy.resources import resource_find
from kivy.logger import Logger
def refresh_fbo_setup(self, *args):
    """(internal) Creates and assigns one :class:`~kivy.graphics.Fbo`
        per effect, and makes sure all sizes etc. are correct and
        consistent.
        """
    while len(self.fbo_list) < len(self.effects):
        with self.canvas:
            new_fbo = EffectFbo(size=self.size)
        with new_fbo:
            ClearColor(0, 0, 0, 0)
            ClearBuffers()
            Color(1, 1, 1, 1)
            new_fbo.texture_rectangle = Rectangle(size=self.size)
            new_fbo.texture_rectangle.size = self.size
        self.fbo_list.append(new_fbo)
    while len(self.fbo_list) > len(self.effects):
        old_fbo = self.fbo_list.pop()
        self.canvas.remove(old_fbo)
    for effect in self._bound_effects:
        if effect not in self.effects:
            effect.fbo = None
    self._bound_effects = self.effects
    self.fbo.size = self.size
    self.fbo_rectangle.size = self.size
    for i in range(len(self.fbo_list)):
        self.fbo_list[i].size = self.size
        self.fbo_list[i].texture_rectangle.size = self.size
    if len(self.fbo_list) == 0:
        self.texture = self.fbo.texture
        return
    for i in range(1, len(self.fbo_list)):
        fbo = self.fbo_list[i]
        fbo.texture_rectangle.texture = self.fbo_list[i - 1].texture
    for effect, fbo in zip(self.effects, self.fbo_list):
        effect.fbo = fbo
    self.fbo_list[0].texture_rectangle.texture = self.fbo.texture
    self.texture = self.fbo_list[-1].texture
    for fbo in self.fbo_list:
        fbo.draw()
    self.fbo.draw()