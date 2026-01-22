import os
import time
from twisted.internet import defer
from twisted.names import common, dns, error
from twisted.python import failure
from twisted.python.compat import execfile, nativeString
from twisted.python.filepath import FilePath
def stripComments(self, lines):
    """
        Strip comments from C{lines}.

        @param lines: lines to work on
        @type lines: iterable of L{bytes}

        @return: C{lines} sans comments.
        """
    return (a.find(b';') == -1 and a or a[:a.find(b';')] for a in [b.strip() for b in lines])