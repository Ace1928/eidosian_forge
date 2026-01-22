from __future__ import unicode_literals
import collections
import contextlib
import inspect
import logging
import pprint
import sys
import textwrap
import six
def has_override(self):
    for descr in self._field_registry:
        if isinstance(descr, SubtreeDescriptor):
            if descr.get(self).has_override():
                return True
        elif isinstance(descr, FieldDescriptor):
            if descr.has_override(self):
                return True
        else:
            raise RuntimeError('Field registry contains unknown descriptor')
    return False