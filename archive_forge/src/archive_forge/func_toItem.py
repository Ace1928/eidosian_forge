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
def toItem(obj):
    if isinstance(obj, ParseResults):
        if obj.haskeys():
            return obj.asDict()
        else:
            return [toItem(v) for v in obj]
    else:
        return obj