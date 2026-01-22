from typing import Any, Tuple, Union
from twisted.application.internet import StreamServerEndpointService
from twisted.cred import error
from twisted.cred.checkers import FilePasswordDB, ICredentialsChecker
from twisted.cred.credentials import ISSHPrivateKey, IUsernamePassword, UsernamePassword
from twisted.internet.defer import Deferred
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
def test_multipleAuthAdded(self) -> None:
    """
        Multiple C{--auth} command-line options will add all checkers specified
        to the list ofcheckers, and there should only be the specified auth
        checkers (no default checkers).
        """
    self.options.parseOptions(['--auth', 'file:' + self.filename, '--auth', 'memory:testuser:testpassword'])
    self.assertEqual(len(self.options['credCheckers']), 2)