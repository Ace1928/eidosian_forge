from __future__ import unicode_literals
import collections
import contextlib
import inspect
import logging
import pprint
import sys
import textwrap
import six
def consume_known(self, kwargs):
    """Consume known configuration values from a dictionary. Removes the
       values from the dictionary.
    """
    for descr in self._field_registry:
        if descr.name in kwargs:
            descr.consume_value(self, kwargs.pop(descr.name))
    self._update_derived()