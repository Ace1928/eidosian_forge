from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def nop(*args, **kw):
    """Do nothing."""