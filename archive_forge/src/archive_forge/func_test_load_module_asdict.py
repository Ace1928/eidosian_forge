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
def test_load_module_asdict():
    with TestNamespace():
        session_buffer = BytesIO()
        dill.dump_module(session_buffer)
        global empty, names, x, y
        x = y = 0
        del empty
        globals_state = globals().copy()
        session_buffer.seek(0)
        main_vars = dill.load_module_asdict(session_buffer)
        assert main_vars is not globals()
        assert globals() == globals_state
        assert main_vars['__name__'] == '__main__'
        assert main_vars['names'] == names
        assert main_vars['names'] is not names
        assert main_vars['x'] != x
        assert 'y' not in main_vars
        assert 'empty' in main_vars