from typing import Any, Tuple, Union
from twisted.application.internet import StreamServerEndpointService
from twisted.cred import error
from twisted.cred.checkers import FilePasswordDB, ICredentialsChecker
from twisted.cred.credentials import ISSHPrivateKey, IUsernamePassword, UsernamePassword
from twisted.internet.defer import Deferred
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
def checkSuccess(username: Union[bytes, Tuple[()]]) -> None:
    self.assertEqual(username, correct.username)