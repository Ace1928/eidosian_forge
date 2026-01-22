from __future__ import unicode_literals
import collections
import copy
import functools
import json
import sys
from jsonpointer import JsonPointer, JsonPointerException
def take_index(self, value, st):
    typed_key = (value, type(value))
    try:
        stored = self.index_storage[st].get(typed_key)
        if stored:
            return stored.pop()
    except TypeError:
        storage = self.index_storage2[st]
        for i in range(len(storage) - 1, -1, -1):
            if storage[i][0] == typed_key:
                return storage.pop(i)[1]