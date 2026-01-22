import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
@expose([wsme.types.bytes])
def getbytesarray(self):
    return [b'A', b'B', b'C']