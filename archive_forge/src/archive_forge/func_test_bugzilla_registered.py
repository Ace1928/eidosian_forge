from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
def test_bugzilla_registered(self):
    """The Bugzilla bug tracker should be registered by default and
        generate Bugzilla bug page URLs when the appropriate configuration is
        present.
        """
    branch = self.make_branch('some_branch')
    config = branch.get_config()
    config.set_user_option('bugzilla_foo_url', 'http://bugs.example.com')
    tracker = bugtracker.tracker_registry.get_tracker('foo', branch)
    self.assertEqual('http://bugs.example.com/show_bug.cgi?id=1234', tracker.get_bug_url('1234'))