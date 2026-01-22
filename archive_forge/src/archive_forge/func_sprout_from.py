from typing import Any, List
from breezy import branchbuilder
from breezy.branch import GenericInterBranch, InterBranch
from breezy.tests import TestCaseWithTransport, multiply_tests
def sprout_from(self, origdir, to_url):
    """Sprout a bzrdir, using from_format for the new bzrdir."""
    wt = self._sprout(origdir, to_url, self.branch_format_from._matchingcontroldir)
    self.assertEqual(wt.branch._format, self.branch_format_from)
    return wt.controldir