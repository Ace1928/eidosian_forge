import sys
from typing import Tuple
import pyglet
import pyglet.window.key
import pyglet.window.mouse
from pyglet import gl
from pyglet.math import Mat4
from pyglet.event import EventDispatcher
from pyglet.window import key, event
from pyglet.graphics import shader
def set_fullscreen(self, fullscreen=True, screen=None, mode=None, width=None, height=None):
    """Toggle to or from fullscreen.

        After toggling fullscreen, the GL context should have retained its
        state and objects, however the buffers will need to be cleared and
        redrawn.

        If `width` and `height` are specified and `fullscreen` is True, the
        screen may be switched to a different resolution that most closely
        matches the given size.  If the resolution doesn't match exactly,
        a higher resolution is selected and the window will be centered
        within a black border covering the rest of the screen.

        :Parameters:
            `fullscreen` : bool
                True if the window should be made fullscreen, False if it
                should be windowed.
            `screen` : Screen
                If not None and fullscreen is True, the window is moved to the
                given screen.  The screen must belong to the same display as
                the window.
            `mode` : `ScreenMode`
                The screen will be switched to the given mode.  The mode must
                have been obtained by enumerating `Screen.get_modes`.  If
                None, an appropriate mode will be selected from the given
                `width` and `height`.
            `width` : int
                Optional width of the window.  If unspecified, defaults to the
                previous window size when windowed, or the screen size if
                fullscreen.

                .. versionadded:: 1.2
            `height` : int
                Optional height of the window.  If unspecified, defaults to
                the previous window size when windowed, or the screen size if
                fullscreen.

                .. versionadded:: 1.2
        """
    if fullscreen == self._fullscreen and (screen is None or screen is self._screen) and (width is None or width == self._width) and (height is None or height == self._height):
        return
    if not self._fullscreen:
        self._windowed_size = self.get_size()
        self._windowed_location = self.get_location()
    if fullscreen and screen is not None:
        assert screen.display is self.display
        self._screen = screen
    self._fullscreen = fullscreen
    if self._fullscreen:
        self._width, self._height = self._set_fullscreen_mode(mode, width, height)
    else:
        self.screen.restore_mode()
        self._width, self._height = self._windowed_size
        if width is not None:
            self._width = width
        if height is not None:
            self._height = height
    self._recreate(['fullscreen'])
    if not self._fullscreen and self._windowed_location:
        self.set_location(*self._windowed_location)