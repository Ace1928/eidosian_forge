import io
import os
import socket
import stat
from hashlib import md5
from typing import IO
from zope.interface import implementer
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, interfaces, reactor
from twisted.mail import mail, pop3, smtp
from twisted.persisted import dirdbm
from twisted.protocols import basic
from twisted.python import failure, log

        Authenticate a user and, if successful, return their username.

        @type c: L{IUsernamePassword <credentials.IUsernamePassword>} or
            L{IUsernameHashedPassword <credentials.IUsernameHashedPassword>}
            provider.
        @param c: Credentials.

        @rtype: L{bytes}
        @return: A string which identifies an user.

        @raise UnauthorizedLogin: When the credentials check fails.
        