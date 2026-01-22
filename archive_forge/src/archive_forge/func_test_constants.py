from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_constants(self):
    if regex.I != regex.IGNORECASE:
        self.fail()
    if regex.L != regex.LOCALE:
        self.fail()
    if regex.M != regex.MULTILINE:
        self.fail()
    if regex.S != regex.DOTALL:
        self.fail()
    if regex.X != regex.VERBOSE:
        self.fail()