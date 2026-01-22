from pyglet.window import key, mouse
from pyglet.libs.darwin.quartzkey import keymap, charmap
from pyglet.libs.darwin import cocoapy, NSPasteboardURLReadingFileURLsOnlyKey, NSLeftShiftKeyMask, NSRightShiftKeyMask, \
from .pyglet_textview import PygletTextView
@PygletView.method('v@')
def mouseMoved_(self, nsevent):
    if self._window._mouse_ignore_motion:
        self._window._mouse_ignore_motion = False
        return
    if not self._window._mouse_in_window:
        return
    x, y = getMousePosition(self, nsevent)
    dx, dy = getMouseDelta(nsevent)
    self._window.dispatch_event('on_mouse_motion', x, y, dx, dy)