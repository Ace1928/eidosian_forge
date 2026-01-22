from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
def test_subscript_operator_out_of_range(self):
    """
        An C{IndexError} is raised if an invalid index is used when retrieving
        data from a sample collection.
        """
    bug1 = dict(id='1', title='Bug #1')
    self.launchpad.bugs = dict(entries=[bug1])
    self.assertRaises(IndexError, lambda: self.launchpad.bugs[2])