import binascii
import errno
import sys
from base64 import decodebytes
from typing import IO, Any, Callable, Iterable, Iterator, Mapping, Optional, Tuple, cast
from zope.interface import Interface, implementer, providedBy
from incremental import Version
from typing_extensions import Literal, Protocol
from twisted.conch import error
from twisted.conch.ssh import keys
from twisted.cred.checkers import ICredentialsChecker
from twisted.cred.credentials import ISSHPrivateKey, IUsernamePassword
from twisted.cred.error import UnauthorizedLogin, UnhandledCredentials
from twisted.internet import defer
from twisted.logger import Logger
from twisted.plugins.cred_unix import verifyCryptedPassword
from twisted.python import failure, reflect
from twisted.python.deprecate import deprecatedModuleAttribute
from twisted.python.filepath import FilePath
from twisted.python.util import runAsEffectiveUser
def getAuthorizedKeys(self, username: bytes) -> Iterable[keys.Key]:
    try:
        passwd = _lookupUser(self._userdb, username)
    except KeyError:
        return ()
    root = FilePath(passwd.pw_dir).child('.ssh')
    files = ['authorized_keys', 'authorized_keys2']
    return _keysFromFilepaths((root.child(f) for f in files), self._parseKey)