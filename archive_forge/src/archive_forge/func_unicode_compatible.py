import sys
from io import TextIOBase
def unicode_compatible(cls):
    cls.__unicode__ = cls.__str__
    cls.__str__ = lambda x: x.__unicode__().encode('utf-8')
    return cls