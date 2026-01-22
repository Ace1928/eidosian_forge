import signal
from pyglet import app
from pyglet.app.base import PlatformEventLoop, EventLoop
from pyglet.libs.darwin import cocoapy, AutoReleasePool, ObjCSubclass, PyObjectEncoding, ObjCInstance, send_super, \
def nsapp_step(self):
    """Used only for CocoaAlternateEventLoop"""
    self._event_loop.idle()
    self.dispatch_posted_events()