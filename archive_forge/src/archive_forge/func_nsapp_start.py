import signal
from pyglet import app
from pyglet.app.base import PlatformEventLoop, EventLoop
from pyglet.libs.darwin import cocoapy, AutoReleasePool, ObjCSubclass, PyObjectEncoding, ObjCInstance, send_super, \
def nsapp_start(self, interval):
    """Used only for CocoaAlternateEventLoop"""
    from pyglet.app import event_loop
    self._event_loop = event_loop

    def term_received(*args):
        if self.timer:
            self.timer.invalidate()
            self.timer = None
        self.nsapp_stop()
    signal.signal(signal.SIGINT, term_received)
    signal.signal(signal.SIGTERM, term_received)
    self.appdelegate = _AppDelegate.alloc().init(self)
    self.NSApp.setDelegate_(self.appdelegate)
    self.timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(interval, self.appdelegate, get_selector('updatePyglet:'), False, True)
    self.NSApp.run()