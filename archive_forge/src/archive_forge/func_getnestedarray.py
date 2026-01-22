import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
@expose([NestedOuter])
def getnestedarray(self):
    return [NestedOuter(), NestedOuter()]