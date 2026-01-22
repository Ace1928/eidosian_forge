from fontTools.misc.textTools import byteord, strjoin, tobytes, tostr
import sys
import os
import string
def begintag(self, _TAG_, *args, **kwargs):
    attrdata = self.stringifyattrs(*args, **kwargs)
    data = '<%s%s>' % (_TAG_, attrdata)
    self._writeraw(data)
    self.stack.append(_TAG_)
    self.indent()