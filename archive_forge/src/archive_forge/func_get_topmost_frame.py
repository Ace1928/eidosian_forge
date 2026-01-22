from _pydevd_bundle.pydevd_constants import (STATE_RUN, PYTHON_SUSPEND, SUPPORT_GEVENT, ForkSafeLock,
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_frame import PyDBFrame
def get_topmost_frame(self, thread):
    """
        Gets the topmost frame for the given thread. Note that it may be None
        and callers should remove the reference to the frame as soon as possible
        to avoid disturbing user code.
        """
    current_frames = _current_frames()
    topmost_frame = current_frames.get(thread.ident)
    if topmost_frame is None:
        pydev_log.info('Unable to get topmost frame for thread: %s, thread.ident: %s, id(thread): %s\nCurrent frames: %s.\nGEVENT_SUPPORT: %s', thread, thread.ident, id(thread), current_frames, SUPPORT_GEVENT)
    return topmost_frame