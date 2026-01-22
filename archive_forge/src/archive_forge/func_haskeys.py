import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import collections
import pprint
import traceback
import types
from datetime import datetime
def haskeys(self):
    """Since keys() returns an iterator, this method is helpful in bypassing
           code that looks for the existence of any defined results names."""
    return bool(self.__tokdict)