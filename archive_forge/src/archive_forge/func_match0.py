import os
import re
import stat
import sys
import fnmatch
from xdg import BaseDirectory
import xdg.Locale
from xml.dom import minidom, XML_NAMESPACE
from collections import defaultdict
def match0(self, buffer):
    l = len(buffer)
    lenvalue = len(self.value)
    for o in range(self.range):
        s = self.start + o
        e = s + lenvalue
        if l < e:
            return False
        if self.mask:
            test = ''
            for i in range(lenvalue):
                if PY3:
                    c = buffer[s + i] & self.mask[i]
                else:
                    c = ord(buffer[s + i]) & ord(self.mask[i])
                test += chr(c)
        else:
            test = buffer[s:e]
        if test == self.value:
            return True