import unittest as pyunit
from twisted.python.util import mergeFunctionMetadata
from twisted.trial import unittest
def goodDecorator(fn):
    """
    Decorate a function and preserve the original name.
    """

    def nameCollision(*args, **kwargs):
        return fn(*args, **kwargs)
    return mergeFunctionMetadata(fn, nameCollision)