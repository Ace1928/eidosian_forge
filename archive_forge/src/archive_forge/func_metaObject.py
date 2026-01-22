import sys
import re
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.misc import Literal, moduleMember
def metaObject(self):

    class _FakeMetaObject(object):

        def className(*args):
            return self.__class__.__name__
    return _FakeMetaObject()