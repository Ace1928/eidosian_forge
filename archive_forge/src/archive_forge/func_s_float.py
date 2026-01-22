from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def s_float(scanner, token):
    return float(token)