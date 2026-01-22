import sys
from pyasn1 import error
from pyasn1.compat import calling
from pyasn1.type import constraint
from pyasn1.type import tag
from pyasn1.type import tagmap
@staticmethod
def isNoValue(*values):
    for value in values:
        if value is not noValue:
            return False
    return True