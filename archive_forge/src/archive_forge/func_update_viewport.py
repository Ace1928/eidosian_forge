from os.path import join, exists
from os import getcwd
from collections import defaultdict
from kivy.core import core_select_lib
from kivy.clock import Clock
from kivy.config import Config
from kivy.logger import Logger
from kivy.base import EventLoop, stopTouchApp
from kivy.modules import Modules
from kivy.event import EventDispatcher
from kivy.properties import ListProperty, ObjectProperty, AliasProperty, \
from kivy.utils import platform, reify, deprecated, pi_version
from kivy.context import get_current_context
from kivy.uix.behaviors import FocusBehavior
from kivy.setupconfig import USE_SDL2
from kivy.graphics.transformation import Matrix
from kivy.graphics.cgl import cgl_get_backend_name
def update_viewport(self):
    from kivy.graphics.opengl import glViewport
    from kivy.graphics.transformation import Matrix
    from math import radians
    w, h = self._get_effective_size()
    smode = self.softinput_mode
    target = self._system_keyboard.target
    targettop = max(0, target.to_window(0, target.y)[1]) if target else 0
    kheight = self._kheight
    w2, h2 = (w / 2.0, h / 2.0)
    r = radians(self.rotation)
    y = 0
    _h = h
    if smode == 'pan':
        y = kheight
    elif smode == 'below_target':
        y = 0 if kheight < targettop else kheight - targettop
    if smode == 'scale':
        _h -= kheight
    glViewport(0, 0, w, _h)
    projection_mat = Matrix()
    projection_mat.view_clip(0.0, w, 0.0, h, -1.0, 1.0, 0)
    self.render_context['projection_mat'] = projection_mat
    modelview_mat = Matrix().translate(w2, h2, 0)
    modelview_mat = modelview_mat.multiply(Matrix().rotate(r, 0, 0, 1))
    w, h = self.size
    w2, h2 = (w / 2.0, h / 2.0 - y)
    modelview_mat = modelview_mat.multiply(Matrix().translate(-w2, -h2, 0))
    self.render_context['modelview_mat'] = modelview_mat
    frag_modelview_mat = Matrix()
    frag_modelview_mat.set(flat=modelview_mat.get())
    self.render_context['frag_modelview_mat'] = frag_modelview_mat
    self.canvas.ask_update()
    self.update_childsize()