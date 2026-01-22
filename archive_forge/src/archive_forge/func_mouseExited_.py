from pyglet.window import key, mouse
from pyglet.libs.darwin.quartzkey import keymap, charmap
from pyglet.libs.darwin import cocoapy, NSPasteboardURLReadingFileURLsOnlyKey, NSLeftShiftKeyMask, NSRightShiftKeyMask, \
from .pyglet_textview import PygletTextView
@PygletView.method('v@')
def mouseExited_(self, nsevent):
    x, y = getMousePosition(self, nsevent)
    self._window._mouse_in_window = False
    if not self._window._mouse_exclusive:
        self._window.set_mouse_platform_visible()
    self._window.dispatch_event('on_mouse_leave', x, y)