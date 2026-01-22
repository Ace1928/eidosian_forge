from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
def test_doesnt_consult_branch(self):
    """Shouldn't consult the branch for tracker information.
        """
    tracker = bugtracker.ProjectIntegerBugTracker('xxx', 'http://bugs.example.com/{project}/{id}')
    self.assertIs(tracker, tracker.get('xxx', None))
    self.assertIs(None, tracker.get('yyy', None))