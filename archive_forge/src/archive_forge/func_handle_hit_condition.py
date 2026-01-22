from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_import_class
from _pydevd_bundle.pydevd_frame_utils import add_exception_to_frame
from _pydev_bundle._pydev_saved_modules import threading
def handle_hit_condition(self, frame):
    if not self.hit_condition:
        return False
    ret = False
    with self._hit_condition_lock:
        self._hit_count += 1
        expr = self.hit_condition.replace('@HIT@', str(self._hit_count))
        try:
            ret = bool(eval(expr, frame.f_globals, frame.f_locals))
        except Exception:
            ret = False
    return ret