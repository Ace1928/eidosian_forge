from fontTools.misc.textTools import byteord, strjoin, tobytes, tostr
import sys
import os
import string
def endtag(self, _TAG_):
    assert self.stack and self.stack[-1] == _TAG_, 'nonmatching endtag'
    del self.stack[-1]
    self.dedent()
    data = '</%s>' % _TAG_
    self._writeraw(data)