from pyglet.window import key, mouse
from pyglet.libs.darwin.quartzkey import keymap, charmap
from pyglet.libs.darwin import cocoapy, NSPasteboardURLReadingFileURLsOnlyKey, NSLeftShiftKeyMask, NSRightShiftKeyMask, \
from .pyglet_textview import PygletTextView
@PygletView.method('v@')
def flagsChanged_(self, nsevent):
    symbol = keymap.get(nsevent.keyCode(), None)
    if symbol is None or symbol not in maskForKey:
        return
    modifiers = getModifiers(nsevent)
    modifierFlags = nsevent.modifierFlags()
    if symbol and modifierFlags & maskForKey[symbol]:
        self._window.dispatch_event('on_key_press', symbol, modifiers)
    else:
        self._window.dispatch_event('on_key_release', symbol, modifiers)