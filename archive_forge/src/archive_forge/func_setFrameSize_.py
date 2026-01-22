from pyglet.window import key, mouse
from pyglet.libs.darwin.quartzkey import keymap, charmap
from pyglet.libs.darwin import cocoapy, NSPasteboardURLReadingFileURLsOnlyKey, NSLeftShiftKeyMask, NSRightShiftKeyMask, \
from .pyglet_textview import PygletTextView
@PygletView.method(b'v' + cocoapy.NSSizeEncoding)
def setFrameSize_(self, size):
    cocoapy.send_super(self, 'setFrameSize:', size, superclass_name='NSView', argtypes=[cocoapy.NSSize])
    if not self._window.context.canvas:
        return
    width, height = (int(size.width), int(size.height))
    self._window.switch_to()
    self._window.context.update_geometry()
    self._window._width, self._window._height = (width, height)
    self._window.dispatch_event('on_resize', width, height)
    self._window.dispatch_event('on_expose')
    if self.inLiveResize():
        from pyglet import app
        if app.event_loop is not None:
            app.event_loop.idle()