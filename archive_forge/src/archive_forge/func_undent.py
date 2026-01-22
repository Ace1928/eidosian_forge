from __future__ import absolute_import
import six
import copy
from collections import OrderedDict
from googleapiclient import _helpers as util
def undent(self):
    """Decrease indentation level."""
    self.dent -= 1