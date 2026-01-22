import sys
from _pydev_bundle import pydev_log
def patched_use(*args, **kwargs):
    matplotlib.real_use(*args, **kwargs)
    gui, backend = find_gui_and_backend()
    enable_gui_function(gui)