import functools
import string
import re
from types import MappingProxyType
from llvmlite.ir import values, types, _utils
from llvmlite.ir._utils import (_StrCaching, _StringReferenceCaching,
def insert_basic_block(self, before, name=''):
    """Insert block before
        """
    blk = Block(parent=self, name=name)
    self.blocks.insert(before, blk)
    return blk