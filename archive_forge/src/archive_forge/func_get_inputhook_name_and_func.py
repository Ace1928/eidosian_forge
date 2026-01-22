import importlib
import os
from typing import Tuple, Callable
def get_inputhook_name_and_func(gui: str) -> Tuple[str, Callable]:
    if gui in registered:
        return (gui, registered[gui])
    if gui not in backends:
        raise UnknownBackend(gui)
    if gui in aliases:
        return get_inputhook_name_and_func(aliases[gui])
    gui_mod = gui
    if gui.startswith('qt'):
        gui = set_qt_api(gui)
        gui_mod = 'qt'
    mod = importlib.import_module('IPython.terminal.pt_inputhooks.' + gui_mod)
    return (gui, mod.inputhook)