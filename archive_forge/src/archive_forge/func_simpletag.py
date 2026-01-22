from fontTools.misc.textTools import byteord, strjoin, tobytes, tostr
import sys
import os
import string
def simpletag(self, _TAG_, *args, **kwargs):
    attrdata = self.stringifyattrs(*args, **kwargs)
    data = '<%s%s/>' % (_TAG_, attrdata)
    self._writeraw(data)