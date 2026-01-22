from __future__ import unicode_literals
import collections
import contextlib
import inspect
import logging
import pprint
import sys
import textwrap
import six
def legacy_shim_consume(self, obj, kwargs):
    """Consume config variable assignments from the root of the config
    tree. This is the legacy config style and will likely be deprecated
    soon."""
    self.get(obj).legacy_consume(kwargs)