import sys
import signal
import time
from timeit import default_timer as clock
import wx
@ignore_keyboardinterrupts
def inputhook_wx3(context):
    """Run the wx event loop by processing pending events only.

    This is like inputhook_wx1, but it keeps processing pending events
    until stdin is ready.  After processing all pending events, a call to
    time.sleep is inserted.  This is needed, otherwise, CPU usage is at 100%.
    This sleep time should be tuned though for best performance.
    """
    app = wx.GetApp()
    if app is not None:
        assert wx.Thread_IsMain()
        if not callable(signal.getsignal(signal.SIGINT)):
            signal.signal(signal.SIGINT, signal.default_int_handler)
        evtloop = wx.EventLoop()
        ea = wx.EventLoopActivator(evtloop)
        t = clock()
        while not context.input_is_ready():
            while evtloop.Pending():
                t = clock()
                evtloop.Dispatch()
            app.ProcessIdle()
            used_time = clock() - t
            if used_time > 10.0:
                time.sleep(1.0)
            elif used_time > 0.1:
                time.sleep(0.05)
            else:
                time.sleep(0.001)
        del ea
    return 0