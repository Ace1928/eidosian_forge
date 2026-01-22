from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six import add_metaclass
@collection_finder.setter
def collection_finder(cls, value):
    if cls._collection_finder:
        raise ValueError('an AnsibleCollectionFinder has already been configured')
    cls._collection_finder = value