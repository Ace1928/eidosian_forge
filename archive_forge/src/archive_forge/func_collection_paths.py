from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six import add_metaclass
@property
def collection_paths(cls):
    cls._require_finder()
    return [to_text(p) for p in cls._collection_finder._n_collection_paths]