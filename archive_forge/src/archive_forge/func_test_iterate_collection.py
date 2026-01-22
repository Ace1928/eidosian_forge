from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
def test_iterate_collection(self):
    """
        Data for a sample collection set on a L{FakeLaunchpad} instance can be
        iterated over if an C{entries} key is defined.
        """
    bug = dict(id='1', title='Bug #1')
    self.launchpad.bugs = dict(entries=[bug])
    bugs = list(self.launchpad.bugs)
    self.assertEqual(1, len(bugs))
    bug = bugs[0]
    self.assertEqual('1', bug.id)
    self.assertEqual('Bug #1', bug.title)