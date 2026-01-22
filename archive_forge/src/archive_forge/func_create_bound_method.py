from __future__ import absolute_import
import functools
import itertools
import operator
import sys
import types
def create_bound_method(func, obj):
    return types.MethodType(func, obj, obj.__class__)