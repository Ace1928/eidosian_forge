from __future__ import absolute_import
import functools
import itertools
import operator
import sys
import types
def create_module(self, spec):
    return self.load_module(spec.name)