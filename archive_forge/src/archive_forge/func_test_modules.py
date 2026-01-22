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
def test_modules(refimported):
    try:
        for obj in ('json', 'url', 'local_mod', 'sax', 'dom'):
            assert globals()[obj].__name__ in sys.modules
        assert 'calendar' in sys.modules and 'cmath' in sys.modules
        import calendar, cmath
        for obj in ('Calendar', 'isleap'):
            assert globals()[obj] is sys.modules['calendar'].__dict__[obj]
        assert __main__.day_name.__module__ == 'calendar'
        if refimported:
            assert __main__.day_name is calendar.day_name
        assert __main__.complex_log is cmath.log
    except AssertionError as error:
        error.args = (_error_line(error, obj, refimported),)
        raise