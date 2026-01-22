import unicodedata
from pyglet.window import key
from pyglet.libs.darwin.cocoapy import ObjCClass, ObjCSubclass, ObjCInstance
from pyglet.libs.darwin.cocoapy import PyObjectEncoding, send_super
from pyglet.libs.darwin.cocoapy import CFSTR, cfstring_to_string, cf
@PygletTextView.method('v@')
def moveToEndOfLineAndModifySelection_(self, sender):
    self._window.dispatch_event('on_text_motion_select', key.MOTION_END_OF_LINE)