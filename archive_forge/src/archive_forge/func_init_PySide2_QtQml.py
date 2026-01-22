from __future__ import print_function, absolute_import
import sys
import struct
import os
from shibokensupport.signature import typing
from shibokensupport.signature.typing import TypeVar, Generic
from shibokensupport.signature.lib.tool import with_metaclass
def init_PySide2_QtQml():
    type_map.update({'PySide2.QtQml.bool volatile': bool, 'QJSValueList()': [], 'QVariantHash()': typing.Dict[str, Variant]})
    return locals()