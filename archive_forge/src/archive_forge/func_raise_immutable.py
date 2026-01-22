from __future__ import absolute_import, print_function
import sys
import copy
from ruamel.yaml.compat import ordereddict, PY2, string_types, MutableSliceableSequence
from ruamel.yaml.scalarstring import ScalarString
from ruamel.yaml.anchor import Anchor
@classmethod
def raise_immutable(cls, *args, **kwargs):
    raise TypeError('{} objects are immutable'.format(cls.__name__))