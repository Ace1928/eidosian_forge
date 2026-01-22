import os
from twisted.application import internet
from twisted.cred import checkers, strcred
from twisted.internet import endpoints
from twisted.mail import alias, mail, maildir, relay, relaymanager
from twisted.python import usage
def opt_maildirdbmdomain(self, domain):
    """
        Generate an SMTP/POP3 virtual domain.

        This option requires an argument of the form 'NAME=PATH' where NAME is
        the DNS domain name for which email will be accepted and where PATH is
        a the filesystem path to a Maildir folder.
        [Example: 'example.com=/tmp/example.com']
        """
    try:
        name, path = domain.split('=')
    except ValueError:
        raise usage.UsageError("Argument to --maildirdbmdomain must be of the form 'name=path'")
    self.last_domain = maildir.MaildirDirdbmDomain(self.service, os.path.abspath(path))
    self.service.addDomain(name, self.last_domain)