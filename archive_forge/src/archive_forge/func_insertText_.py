import unicodedata
from pyglet.window import key
from pyglet.libs.darwin.cocoapy import ObjCClass, ObjCSubclass, ObjCInstance
from pyglet.libs.darwin.cocoapy import PyObjectEncoding, send_super
from pyglet.libs.darwin.cocoapy import CFSTR, cfstring_to_string, cf
@PygletTextView.method('v@')
def insertText_(self, text):
    text = cfstring_to_string(text)
    self.setString_(self.empty_string)
    if text:
        if unicodedata.category(text[0]) != 'Cc':
            self._window.dispatch_event('on_text', text)