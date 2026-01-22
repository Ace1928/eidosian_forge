import sys
import cgi
import time
import traceback
from io import StringIO
from thread import get_ident
from paste import httpexceptions
from paste.request import construct_url, parse_formvars
from paste.util.template import HTMLTemplate, bunch
def traceback_thread(thread_id):
    """
    Returns a plain-text traceback of the given thread, or None if it
    can't get a traceback.
    """
    if not hasattr(sys, '_current_frames'):
        return None
    frames = sys._current_frames()
    if not thread_id in frames:
        return None
    frame = frames[thread_id]
    out = StringIO()
    traceback.print_stack(frame, file=out)
    return out.getvalue()