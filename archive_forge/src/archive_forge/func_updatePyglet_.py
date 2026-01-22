import signal
from pyglet import app
from pyglet.app.base import PlatformEventLoop, EventLoop
from pyglet.libs.darwin import cocoapy, AutoReleasePool, ObjCSubclass, PyObjectEncoding, ObjCInstance, send_super, \
@_AppDelegate.method('v')
def updatePyglet_(self):
    self._pyglet_loop.nsapp_step()