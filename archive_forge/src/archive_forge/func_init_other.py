from __future__ import print_function, absolute_import
import sys
import struct
import os
from shibokensupport.signature import typing
from shibokensupport.signature.typing import TypeVar, Generic
from shibokensupport.signature.lib.tool import with_metaclass
def init_other():
    import numbers
    type_map.update({'other.ExtendsNoImplicitConversion': Missing('other.ExtendsNoImplicitConversion'), 'other.Number': numbers.Number})
    return locals()