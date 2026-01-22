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
def ftp_TYPE(self, type):
    """
        REPRESENTATION TYPE (TYPE)

        The argument specifies the representation type as described
        in the Section on Data Representation and Storage.  Several
        types take a second parameter.  The first parameter is
        denoted by a single Telnet character, as is the second
        Format parameter for ASCII and EBCDIC; the second parameter
        for local byte is a decimal integer to indicate Bytesize.
        The parameters are separated by a <SP> (Space, ASCII code
        32).
        """
    p = type.upper()
    if p:
        f = getattr(self, 'type_' + p[0], None)
        if f is not None:
            return f(p[1:])
        return self.type_UNKNOWN(p)
    return (SYNTAX_ERR,)