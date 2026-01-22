from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
def test_trac_registered(self):
    """The Trac bug tracker should be registered by default and generate
        Trac bug page URLs when the appropriate configuration is present.
        """
    branch = self.make_branch('some_branch')
    config = branch.get_config()
    config.set_user_option('trac_foo_url', 'http://bugs.example.com/trac')
    tracker = bugtracker.tracker_registry.get_tracker('foo', branch)
    self.assertEqual('http://bugs.example.com/trac/ticket/1234', tracker.get_bug_url('1234'))