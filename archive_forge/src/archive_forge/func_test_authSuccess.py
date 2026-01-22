from typing import Any, Tuple, Union
from twisted.application.internet import StreamServerEndpointService
from twisted.cred import error
from twisted.cred.checkers import FilePasswordDB, ICredentialsChecker
from twisted.cred.credentials import ISSHPrivateKey, IUsernamePassword, UsernamePassword
from twisted.internet.defer import Deferred
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
def test_authSuccess(self) -> Deferred[None]:
    """
        The checker created by the C{--auth} command-line option returns a
        L{Deferred} that returns the avatar id when presented with credentials
        that are known to that checker.
        """
    self.options.parseOptions(['--auth', 'file:' + self.filename])
    checker: ICredentialsChecker = self.options['credCheckers'][-1]
    correct = UsernamePassword(*self.usernamePassword)
    d = checker.requestAvatarId(correct)

    def checkSuccess(username: Union[bytes, Tuple[()]]) -> None:
        self.assertEqual(username, correct.username)
    return d.addCallback(checkSuccess)