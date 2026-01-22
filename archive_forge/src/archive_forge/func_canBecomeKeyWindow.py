from ctypes import c_void_p, c_bool
from pyglet.libs.darwin.cocoapy import ObjCClass, ObjCSubclass, send_super
from pyglet.libs.darwin.cocoapy import NSUInteger, NSUIntegerEncoding
from pyglet.libs.darwin.cocoapy import NSRectEncoding
@PygletWindow.method('B')
def canBecomeKeyWindow(self):
    return True