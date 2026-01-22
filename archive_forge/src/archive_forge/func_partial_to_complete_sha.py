from gitdb.db.base import (
from gitdb.util import LazyMixin
from gitdb.exc import (
from gitdb.pack import PackEntity
from functools import reduce
import os
import glob
def partial_to_complete_sha(self, partial_binsha, canonical_length):
    """:return: 20 byte sha as inferred by the given partial binary sha
        :param partial_binsha: binary sha with less than 20 bytes
        :param canonical_length: length of the corresponding canonical representation.
            It is required as binary sha's cannot display whether the original hex sha
            had an odd or even number of characters
        :raise AmbiguousObjectName:
        :raise BadObject: """
    candidate = None
    for item in self._entities:
        item_index = item[1].index().partial_sha_to_index(partial_binsha, canonical_length)
        if item_index is not None:
            sha = item[1].index().sha(item_index)
            if candidate and candidate != sha:
                raise AmbiguousObjectName(partial_binsha)
            candidate = sha
    if candidate:
        return candidate
    raise BadObject(partial_binsha)