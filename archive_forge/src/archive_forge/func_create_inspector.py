import weakref
from functools import partial
from itertools import chain
from kivy.animation import Animation
from kivy.logger import Logger
from kivy.graphics.transformation import Matrix
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.factory import Factory
from kivy.weakproxy import WeakProxy
from kivy.properties import (
def create_inspector(win, ctx, *largs):
    """Create an Inspector instance attached to the *ctx* and bound to the
    Window's :meth:`~kivy.core.window.WindowBase.on_keyboard` event for
    capturing the keyboard shortcut.

        :Parameters:
            `win`: A :class:`Window <kivy.core.window.WindowBase>`
                The application Window to bind to.
            `ctx`: A :class:`~kivy.uix.widget.Widget` or subclass
                The Widget to be inspected.

    """
    ctx.inspector = Inspector(win=win)
    win.bind(children=ctx.inspector.on_window_children, on_keyboard=ctx.inspector.keyboard_shortcut)