import os
import pickle
import sys
from zope.interface import Interface, implementer
from twisted.persisted import styles
from twisted.python import log, runtime
def loadValueFromFile(filename, variable):
    """Load the value of a variable in a Python file.

    Run the contents of the file in a namespace and return the result of the
    variable named C{variable}.

    @param filename: string
    @param variable: string
    """
    with open(filename) as fileObj:
        data = fileObj.read()
    d = {'__file__': filename}
    codeObj = compile(data, filename, 'exec')
    eval(codeObj, d, d)
    value = d[variable]
    return value