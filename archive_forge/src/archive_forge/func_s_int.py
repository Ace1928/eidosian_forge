from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def s_int(scanner, token):
    return int(token)