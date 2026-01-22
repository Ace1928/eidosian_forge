from __future__ import print_function, absolute_import
import sys
import struct
import os
from shibokensupport.signature import typing
from shibokensupport.signature.typing import TypeVar, Generic
from shibokensupport.signature.lib.tool import with_metaclass
def init_PySide2_QtXmlPatterns():
    from PySide2.QtXmlPatterns import QXmlName
    type_map.update({'QXmlName.NamespaceCode': Missing('PySide2.QtXmlPatterns.QXmlName.NamespaceCode'), 'QXmlName.PrefixCode': Missing('PySide2.QtXmlPatterns.QXmlName.PrefixCode')})
    return locals()