import os
from twisted.application import internet
from twisted.cred import checkers, strcred
from twisted.internet import endpoints
from twisted.mail import alias, mail, maildir, relay, relaymanager
from twisted.python import usage
def opt_user(self, user_pass):
    """
        Add a user and password to the last specified domain.
        """
    try:
        user, password = user_pass.split('=', 1)
    except ValueError:
        raise usage.UsageError("Argument to --user must be of the form 'user=password'")
    if self.last_domain:
        self.last_domain.addUser(user, password)
    else:
        raise usage.UsageError('Specify a domain before specifying users')