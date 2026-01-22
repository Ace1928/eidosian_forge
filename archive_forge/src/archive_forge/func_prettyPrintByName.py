from __future__ import absolute_import
import six
import copy
from collections import OrderedDict
from googleapiclient import _helpers as util
def prettyPrintByName(self, name):
    """Get pretty printed object prototype from the schema name.

    Args:
      name: string, Name of schema in the discovery document.

    Returns:
      string, A string that contains a prototype object with
        comments that conforms to the given schema.
    """
    return self._prettyPrintByName(name, seen=[], dent=0)[:-2]