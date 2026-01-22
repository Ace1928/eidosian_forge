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
def extract_tb(tb, limit=0):
    frames = traceback.extract_tb(tb, limit=limit)
    frame_summary = frames[-1]
    return [frame_summary[:2]]