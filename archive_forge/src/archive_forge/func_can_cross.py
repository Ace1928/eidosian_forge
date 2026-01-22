from abc import ABCMeta, abstractmethod
from functools import total_ordering
from nltk.internals import raise_unorderable_types
def can_cross(self):
    return '.' not in self._restrs