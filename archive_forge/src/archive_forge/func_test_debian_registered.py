from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
def test_debian_registered(self):
    """The Debian bug tracker should be registered by default and generate
        bugs.debian.org bug page URLs.
        """
    branch = self.make_branch('some_branch')
    tracker = bugtracker.tracker_registry.get_tracker('deb', branch)
    self.assertEqual('http://bugs.debian.org/1234', tracker.get_bug_url('1234'))