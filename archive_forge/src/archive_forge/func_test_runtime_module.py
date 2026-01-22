import atexit
import os
import sys
import __main__
from contextlib import suppress
from io import BytesIO
import dill
import json                                         # top-level module
import urllib as url                                # top-level module under alias
from xml import sax                                 # submodule
import xml.dom.minidom as dom                       # submodule under alias
import test_dictviews as local_mod                  # non-builtin top-level module
from calendar import Calendar, isleap, day_name     # class, function, other object
from cmath import log as complex_log                # imported with alias
def test_runtime_module():
    from types import ModuleType
    modname = '__runtime__'
    runtime = ModuleType(modname)
    runtime.x = 42
    mod = dill.session._stash_modules(runtime)
    if mod is not runtime:
        print("There are objects to save by referenece that shouldn't be:", mod.__dill_imported, mod.__dill_imported_as, mod.__dill_imported_top_level, file=sys.stderr)
    session_buffer = BytesIO()
    dill.dump_module(session_buffer, module=runtime, refimported=True)
    session_dump = session_buffer.getvalue()
    runtime = ModuleType(modname)
    return_val = dill.load_module(BytesIO(session_dump), module=runtime)
    assert return_val is None
    assert runtime.__name__ == modname
    assert runtime.x == 42
    assert runtime not in sys.modules.values()
    session_buffer.seek(0)
    runtime = dill.load_module(BytesIO(session_dump))
    assert runtime.__name__ == modname
    assert runtime.x == 42
    assert runtime not in sys.modules.values()