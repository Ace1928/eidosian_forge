import os
import sys
from io import StringIO
from unittest import skipIf
from twisted.copyright import version
from twisted.internet.defer import Deferred
from twisted.internet.testing import MemoryReactor
from twisted.mail import smtp
from twisted.mail.scripts import mailmail
from twisted.mail.scripts.mailmail import parseOptions
from twisted.python.failure import Failure
from twisted.python.runtime import platformType
from twisted.trial.unittest import TestCase
@skipIf(platformType == 'win32', 'mailmail.run() does not work on win32 due to lack of support for getuid()')
def test_readInvalidConfig(self):
    """
        Error messages for illegal UID value, illegal GID value, and illegal
        identity entry will be sent to stderr.
        """
    stdin = StringIO('\n')
    self.patch(sys, 'stdin', stdin)
    filename = self.mktemp()
    myUid = os.getuid()
    myGid = os.getgid()
    with open(filename, 'w') as f:
        f.write('[useraccess]\nallow=invaliduser2,invaliduser1\ndeny=invaliduser3,invaliduser4,{}\norder=allow,deny\n[groupaccess]\nallow=invalidgid1,invalidgid2\ndeny=invalidgid1,invalidgid2,{}\norder=deny,allow\n[identity]\nlocalhost=funny\n[addresses]\nsmarthost=localhost\ndefault_domain=example.com\n'.format(myUid, myGid))
    self.patch(mailmail, 'LOCAL_CFG', filename)
    argv = ('test_mailmail.py', 'invaliduser2@example.com', '-oep')
    self.patch(sys, 'argv', argv)
    mailmail.run()
    self.assertRegex(self.out.getvalue(), 'Illegal UID in \\[useraccess\\] section: invaliduser1')
    self.assertRegex(self.out.getvalue(), 'Illegal GID in \\[groupaccess\\] section: invalidgid1')
    self.assertRegex(self.out.getvalue(), 'Illegal entry in \\[identity\\] section: funny')