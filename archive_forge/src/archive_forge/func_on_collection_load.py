from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six import add_metaclass
@on_collection_load.setter
def on_collection_load(cls, value):
    if value is not cls._on_collection_load:
        raise ValueError('on_collection_load is not directly settable (use +=)')