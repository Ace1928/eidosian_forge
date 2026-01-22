import errno
import fnmatch
import os
import re
import stat
import time
from zope.interface import Interface, implementer
from twisted import copyright
from twisted.cred import checkers, credentials, error as cred_error, portal
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.protocols import basic, policies
from twisted.python import failure, filepath, log
def parseDirectoryLine(self, line):
    """
        Return a dictionary of fields, or None if line cannot be parsed.

        @param line: line of text expected to contain a directory entry
        @type line: str

        @return: dict
        """
    match = self.fileLinePattern.match(line)
    if match is None:
        return None
    else:
        d = match.groupdict()
        d['filename'] = d['filename'].replace('\\ ', ' ')
        d['nlinks'] = int(d['nlinks'])
        d['size'] = int(d['size'])
        if d['linktarget']:
            d['linktarget'] = d['linktarget'].replace('\\ ', ' ')
        return d