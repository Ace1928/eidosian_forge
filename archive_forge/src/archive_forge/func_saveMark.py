from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def saveMark():
    return (lineno, colno)