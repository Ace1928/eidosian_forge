from _pydevd_bundle.pydevd_constants import get_current_thread_id, Null, ForkSafeLock
from pydevd_file_utils import get_abs_path_real_path_and_base_from_frame
from _pydev_bundle._pydev_saved_modules import thread, threading
import sys
from _pydev_bundle import pydev_log
def update_custom_frame(frame_custom_thread_id, frame, thread_id, name=None):
    with CustomFramesContainer.custom_frames_lock:
        if DEBUG:
            sys.stderr.write('update_custom_frame: %s\n' % frame_custom_thread_id)
        try:
            old = CustomFramesContainer.custom_frames[frame_custom_thread_id]
            if name is not None:
                old.name = name
            old.mod_time += 1
            old.thread_id = thread_id
        except:
            sys.stderr.write('Unable to get frame to replace: %s\n' % (frame_custom_thread_id,))
            pydev_log.exception()
        CustomFramesContainer._py_db_command_thread_event.set()