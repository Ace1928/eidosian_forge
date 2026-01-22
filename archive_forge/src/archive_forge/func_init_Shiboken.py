from __future__ import print_function, absolute_import
import sys
import struct
import os
from shibokensupport.signature import typing
from shibokensupport.signature.typing import TypeVar, Generic
from shibokensupport.signature.lib.tool import with_metaclass
def init_Shiboken():
    type_map.update({'PyType': type, 'shiboken2.bool': bool, 'size_t': int})
    return locals()