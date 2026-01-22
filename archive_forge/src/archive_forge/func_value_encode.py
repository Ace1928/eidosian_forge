import re
import string
import types
def value_encode(self, val):
    strval = str(val)
    return (strval, _quote(strval))