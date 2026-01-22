import sys, re, traceback, threading
from .. import exceptionHandling as exceptionHandling
from ..Qt import QtWidgets, QtCore
from ..functions import SignalBlock
from .stackwidget import StackWidget
def checkException(self, excType, exc, tb):
    filename = tb.tb_frame.f_code.co_filename
    function = tb.tb_frame.f_code.co_name
    filterStr = self.filterString
    if filterStr != '':
        if isinstance(exc, Exception):
            msg = traceback.format_exception_only(type(exc), exc)
        elif isinstance(exc, str):
            msg = exc
        else:
            msg = repr(exc)
        match = re.search(filterStr, '%s:%s:%s' % (filename, function, msg))
        return match is not None
    if excType is GeneratorExit or excType is StopIteration:
        return False
    if excType is AttributeError:
        if filename.endswith('numpy/core/fromnumeric.py') and function in ('all', '_wrapit', 'transpose', 'sum'):
            return False
        if filename.endswith('numpy/core/arrayprint.py') and function in '_array2string':
            return False
        if filename.endswith('MetaArray.py') and function == '__getattr__':
            for name in ('__array_interface__', '__array_struct__', '__array__'):
                if name in exc:
                    return False
        if filename.endswith('flowchart/eq.py'):
            return False
    if excType is TypeError:
        if filename.endswith('numpy/lib/function_base.py') and function == 'iterable':
            return False
    return True