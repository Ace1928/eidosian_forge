import collections
import functools
import operator
from ovs.db import data
def index_create(self, name):
    if name in self.indexes:
        raise ValueError('An index named {} already exists'.format(name))
    index = self.indexes[name] = MultiColumnIndex(name)
    return index