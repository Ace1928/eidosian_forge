from _pydev_bundle._pydev_saved_modules import thread, _code
from _pydevd_bundle.pydevd_constants import IS_JYTHON
from _pydevd_bundle.pydevconsole_code import InteractiveConsole
import os
import sys
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_constants import INTERACTIVE_MODE_AVAILABLE
import traceback
from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_save_locals
from _pydev_bundle.pydev_imports import Exec, _queue
import builtins as __builtin__
from _pydev_bundle.pydev_console_utils import BaseInterpreterInterface, BaseStdIn  # @UnusedImport
from _pydev_bundle.pydev_console_utils import CodeFragment
from _pydev_bundle.pydev_umd import runfile, _set_globals_function
def init_mpl_in_console(interpreter):
    init_set_return_control_back(interpreter)
    if not INTERACTIVE_MODE_AVAILABLE:
        return
    activate_mpl_if_already_imported(interpreter)
    from _pydev_bundle.pydev_import_hook import import_hook_manager
    for mod in list(interpreter.mpl_modules_for_patching):
        import_hook_manager.add_module_name(mod, interpreter.mpl_modules_for_patching.pop(mod))