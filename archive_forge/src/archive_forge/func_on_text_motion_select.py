import pyglet
from pyglet.event import EventDispatcher
from pyglet.graphics import Group
from pyglet.text.caret import Caret
from pyglet.text.layout import IncrementalTextLayout
def on_text_motion_select(self, motion):
    if not self.enabled:
        return
    if self._focus:
        self._caret.on_text_motion_select(motion)