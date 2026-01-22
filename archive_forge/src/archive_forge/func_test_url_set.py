from unittest import TestLoader
from .... import config, tests
from ....bzr.bzrdir import BzrDir
from ....tests import TestCaseInTempDir
from ..emailer import EmailSender
def test_url_set(self):
    sender, revid = self.get_sender(with_url_config)
    self.assertEqual(sender.url(), 'http://some.fake/url/')