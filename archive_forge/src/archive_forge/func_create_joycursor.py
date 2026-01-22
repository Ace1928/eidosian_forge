from kivy.clock import Clock
from kivy.logger import Logger
from kivy.uix.widget import Widget
from kivy.graphics import Color, Line
from kivy.properties import (
def create_joycursor(win, ctx, *args):
    """Create a JoyCursor instance attached to the *ctx* and bound to the
    Window's :meth:`~kivy.core.window.WindowBase.on_keyboard` event for
    capturing the keyboard shortcuts.

        :Parameters:
            `win`: A :class:`Window <kivy.core.window.WindowBase>`
                The application Window to bind to.
            `ctx`: A :class:`~kivy.uix.widget.Widget` or subclass
                The Widget for JoyCursor to attach to.

    """
    ctx.joycursor = JoyCursor(win=win)
    win.bind(children=ctx.joycursor.on_window_children, on_keyboard=ctx.joycursor.keyboard_shortcuts)
    win.fbind('on_joy_button_down', ctx.joycursor.joystick_shortcuts)