from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_import_class
from _pydevd_bundle.pydevd_frame_utils import add_exception_to_frame
from _pydev_bundle._pydev_saved_modules import threading
def get_exception_class(kls):
    try:
        return eval(kls)
    except:
        return pydevd_import_class.import_name(kls)