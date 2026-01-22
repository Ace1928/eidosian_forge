from pyglet.libs.darwin.cocoapy import ObjCClass, ObjCSubclass, ObjCInstance
from pyglet.libs.darwin.cocoapy import NSApplicationDidHideNotification
from pyglet.libs.darwin.cocoapy import NSApplicationDidUnhideNotification
from pyglet.libs.darwin.cocoapy import send_super, get_selector
from pyglet.libs.darwin.cocoapy import PyObjectEncoding
from pyglet.libs.darwin.cocoapy import quartz
from .systemcursor import SystemCursor
@PygletDelegate.method(b'@' + PyObjectEncoding)
def initWithWindow_(self, window):
    self = ObjCInstance(send_super(self, 'init'))
    if not self:
        return None
    self._window = window
    window._nswindow.setDelegate_(self)
    notificationCenter = NSNotificationCenter.defaultCenter()
    notificationCenter.addObserver_selector_name_object_(self, get_selector('applicationDidHide:'), NSApplicationDidHideNotification, None)
    notificationCenter.addObserver_selector_name_object_(self, get_selector('applicationDidUnhide:'), NSApplicationDidUnhideNotification, None)
    self.did_pause_exclusive_mouse = False
    return self