from pyglet.window import key, mouse
from pyglet.libs.darwin.quartzkey import keymap, charmap
from pyglet.libs.darwin import cocoapy, NSPasteboardURLReadingFileURLsOnlyKey, NSLeftShiftKeyMask, NSRightShiftKeyMask, \
from .pyglet_textview import PygletTextView
@PygletView.method('v')
def updateTrackingAreas(self):
    if self._tracking_area:
        self.removeTrackingArea_(self._tracking_area)
        self._tracking_area.release()
        self._tracking_area = None
    tracking_options = cocoapy.NSTrackingMouseEnteredAndExited | cocoapy.NSTrackingActiveInActiveApp | cocoapy.NSTrackingCursorUpdate
    frame = self.frame()
    self._tracking_area = NSTrackingArea.alloc().initWithRect_options_owner_userInfo_(frame, tracking_options, self, None)
    self.addTrackingArea_(self._tracking_area)