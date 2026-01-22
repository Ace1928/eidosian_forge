from unittest import TestLoader
from .... import config, tests
from ....bzr.bzrdir import BzrDir
from ....tests import TestCaseInTempDir
from ..emailer import EmailSender
def get_sender(self, text=sample_config):
    my_config = config.MemoryStack(text)
    self.branch = BzrDir.create_branch_convenience('.')
    tree = self.branch.controldir.open_workingtree()
    revid = tree.commit('foo bar baz\nfuzzy\nwuzzy', allow_pointless=True, timestamp=1, timezone=0, committer='Sample <john@example.com>')
    sender = EmailSender(self.branch, revid, my_config)
    sender._setup_revision_and_revno()
    return (sender, revid)