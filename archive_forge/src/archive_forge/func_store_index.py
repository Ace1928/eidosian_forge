from __future__ import unicode_literals
import collections
import copy
import functools
import json
import sys
from jsonpointer import JsonPointer, JsonPointerException
def store_index(self, value, index, st):
    typed_key = (value, type(value))
    try:
        storage = self.index_storage[st]
        stored = storage.get(typed_key)
        if stored is None:
            storage[typed_key] = [index]
        else:
            storage[typed_key].append(index)
    except TypeError:
        self.index_storage2[st].append((typed_key, index))