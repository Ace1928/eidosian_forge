import copy
import sys
import re
import os
from itertools import chain
from contextlib import contextmanager
from parso.python import tree
@contextmanager
def reraise_getitem_errors(*exception_classes):
    try:
        yield
    except exception_classes as e:
        raise SimpleGetItemNotFound(e)