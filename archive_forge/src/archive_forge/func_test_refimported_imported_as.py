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
def test_refimported_imported_as():
    import collections
    import concurrent.futures
    import types
    import typing
    mod = sys.modules['__test__'] = types.ModuleType('__test__')
    dill.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    mod.Dict = collections.UserDict
    mod.AsyncCM = typing.AsyncContextManager
    mod.thread_exec = dill.executor
    session_buffer = BytesIO()
    dill.dump_module(session_buffer, mod, refimported=True)
    session_buffer.seek(0)
    mod = dill.load(session_buffer)
    del sys.modules['__test__']
    assert set(mod.__dill_imported_as) == {('collections', 'UserDict', 'Dict'), ('typing', 'AsyncContextManager', 'AsyncCM'), ('dill', 'executor', 'thread_exec')}