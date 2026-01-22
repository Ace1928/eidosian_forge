import sys, traceback
from ..Qt import QtWidgets, QtGui
def exceptionChain(exc):
    """Return a list of (exception, 'cause'|'context') pairs for exceptions
    leading up to *exc*
    """
    exceptions = [(exc, None)]
    while True:
        if exc.__cause__ is not None:
            exc = exc.__cause__
            exceptions.insert(0, (exc, 'cause'))
        elif exc.__context__ is not None and exc.__suppress_context__ is False:
            exc = exc.__context__
            exceptions.insert(0, (exc, 'context'))
        else:
            break
    return exceptions