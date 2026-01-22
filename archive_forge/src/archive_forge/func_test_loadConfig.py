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
def test_loadConfig(self):
    """
        L{twisted.mail.scripts.mailmail.loadConfig}
        parses the config file for mailmail.
        """
    config = self.getConfigFromFile('\n[addresses]\nsmarthost=localhost')
    self.assertEqual(config.smarthost, 'localhost')
    config = self.getConfigFromFile('\n[addresses]\ndefault_domain=example.com')
    self.assertEqual(config.domain, 'example.com')
    config = self.getConfigFromFile('\n[addresses]\nsmarthost=localhost\ndefault_domain=example.com')
    self.assertEqual(config.smarthost, 'localhost')
    self.assertEqual(config.domain, 'example.com')
    config = self.getConfigFromFile('\n[identity]\nhost1=invalid\nhost2=username:password')
    self.assertNotIn('host1', config.identities)
    self.assertEqual(config.identities['host2'], ['username', 'password'])
    config = self.getConfigFromFile('\n[useraccess]\nallow=invalid1,35\norder=allow')
    self.assertEqual(config.allowUIDs, [35])
    config = self.getConfigFromFile('\n[useraccess]\ndeny=35,36\norder=deny')
    self.assertEqual(config.denyUIDs, [35, 36])
    config = self.getConfigFromFile('\n[useraccess]\nallow=35,36\ndeny=37,38\norder=deny')
    self.assertEqual(config.allowUIDs, [35, 36])
    self.assertEqual(config.denyUIDs, [37, 38])
    config = self.getConfigFromFile('\n[groupaccess]\nallow=gid1,41\norder=allow')
    self.assertEqual(config.allowGIDs, [41])
    config = self.getConfigFromFile('\n[groupaccess]\ndeny=41\norder=deny')
    self.assertEqual(config.denyGIDs, [41])
    config = self.getConfigFromFile('\n[groupaccess]\nallow=41,42\ndeny=43,44\norder=allow,deny')
    self.assertEqual(config.allowGIDs, [41, 42])
    self.assertEqual(config.denyGIDs, [43, 44])