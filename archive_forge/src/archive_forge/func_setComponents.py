import sys
from pyasn1 import error
from pyasn1.compat import calling
from pyasn1.type import constraint
from pyasn1.type import tag
from pyasn1.type import tagmap
def setComponents(self, *args, **kwargs):
    for idx, value in enumerate(args):
        self[idx] = value
    for k in kwargs:
        self[k] = kwargs[k]
    return self