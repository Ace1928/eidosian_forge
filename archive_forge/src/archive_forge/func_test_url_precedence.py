from unittest import TestLoader
from .... import config, tests
from ....bzr.bzrdir import BzrDir
from ....tests import TestCaseInTempDir
from ..emailer import EmailSender
def test_url_precedence(self):
    config = b'[DEFAULT]\npost_commit_url=http://some.fake/url/\npublic_branch=http://the.publication/location/\n'
    sender, revid = self.get_sender(config)
    self.assertEqual(sender.url(), 'http://some.fake/url/')