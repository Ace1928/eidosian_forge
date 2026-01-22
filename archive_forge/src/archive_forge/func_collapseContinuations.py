import os
import time
from twisted.internet import defer
from twisted.names import common, dns, error
from twisted.python import failure
from twisted.python.compat import execfile, nativeString
from twisted.python.filepath import FilePath
def collapseContinuations(self, lines):
    """
        Transform multiline statements into single lines.

        @param lines: lines to work on
        @type lines: iterable of L{bytes}

        @return: iterable of continuous lines
        """
    l = []
    state = 0
    for line in lines:
        if state == 0:
            if line.find(b'(') == -1:
                l.append(line)
            else:
                l.append(line[:line.find(b'(')])
                state = 1
        elif line.find(b')') != -1:
            l[-1] += b' ' + line[:line.find(b')')]
            state = 0
        else:
            l[-1] += b' ' + line
    return filter(None, (line.split() for line in l))