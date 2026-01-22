from __future__ import absolute_import
import six
import copy
from collections import OrderedDict
from googleapiclient import _helpers as util
def emitBegin(self, text):
    """Add text to the output, but with no line terminator.

    Args:
      text: string, Text to output.
      """
    self.value.extend(['  ' * self.dent, text])