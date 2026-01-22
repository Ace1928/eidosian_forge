import os
from twisted.application import internet
from twisted.cred import checkers, strcred
from twisted.internet import endpoints
from twisted.mail import alias, mail, maildir, relay, relaymanager
from twisted.python import usage
def opt_pop3(self, description):
    """
        Add a POP3 port listener on the specified endpoint.

        You can listen on multiple ports by specifying multiple --pop3 options.
        """
    self.addEndpoint('pop3', description)