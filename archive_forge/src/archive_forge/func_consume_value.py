from __future__ import unicode_literals
import collections
import contextlib
import inspect
import logging
import pprint
import sys
import textwrap
import six
def consume_value(self, obj, value):
    """Convenience method to consume values directly from the descriptor
    interface."""
    self.__set__(obj, value)