from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
def printX(*args, **kwargs):
    raise Exception('Test coding error: using print() directly, should use print_()')