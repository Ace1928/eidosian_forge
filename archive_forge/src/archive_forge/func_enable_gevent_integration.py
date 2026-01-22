import pydevd_tracing
import greenlet
import gevent
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_custom_frames import add_custom_frame, update_custom_frame, remove_custom_frame
from _pydevd_bundle.pydevd_constants import GEVENT_SHOW_PAUSED_GREENLETS, get_global_debugger, \
from _pydev_bundle import pydev_log
from pydevd_file_utils import basename
def enable_gevent_integration():
    try:
        if tuple((int(x) for x in gevent.__version__.split('.')[:2])) <= (20, 0):
            if not GEVENT_SHOW_PAUSED_GREENLETS:
                return
            if not hasattr(greenlet, 'settrace'):
                pydev_log.debug('greenlet.settrace not available. GEVENT_SHOW_PAUSED_GREENLETS will have no effect.')
                return
        try:
            greenlet.settrace(greenlet_events)
        except:
            pydev_log.exception('Error with greenlet.settrace.')
    except:
        pydev_log.exception('Error setting up gevent %s.', gevent.__version__)