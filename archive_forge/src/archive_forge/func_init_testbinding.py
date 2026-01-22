from __future__ import print_function, absolute_import
import sys
import struct
import os
from shibokensupport.signature import typing
from shibokensupport.signature.typing import TypeVar, Generic
from shibokensupport.signature.lib.tool import with_metaclass
def init_testbinding():
    type_map.update({'testbinding.PySideCPP2.TestObjectWithoutNamespace': testbinding.TestObjectWithoutNamespace})
    return locals()