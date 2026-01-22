from pyglet.window import key, mouse
from pyglet.libs.darwin.quartzkey import keymap, charmap
from pyglet.libs.darwin import cocoapy, NSPasteboardURLReadingFileURLsOnlyKey, NSLeftShiftKeyMask, NSRightShiftKeyMask, \
from .pyglet_textview import PygletTextView
@PygletView.method('B@')
def performKeyEquivalent_(self, nsevent):
    modifierFlags = nsevent.modifierFlags()
    if modifierFlags & cocoapy.NSNumericPadKeyMask:
        return False
    if modifierFlags & cocoapy.NSFunctionKeyMask:
        ch = cocoapy.cfstring_to_string(nsevent.charactersIgnoringModifiers())
        if ch in (cocoapy.NSHomeFunctionKey, cocoapy.NSEndFunctionKey, cocoapy.NSPageUpFunctionKey, cocoapy.NSPageDownFunctionKey):
            return False
    NSApp = cocoapy.ObjCClass('NSApplication').sharedApplication()
    NSApp.mainMenu().performKeyEquivalent_(nsevent)
    return True