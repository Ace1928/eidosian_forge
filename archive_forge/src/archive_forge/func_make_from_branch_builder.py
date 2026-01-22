from typing import Any, List
from breezy import branchbuilder
from breezy.branch import GenericInterBranch, InterBranch
from breezy.tests import TestCaseWithTransport, multiply_tests
def make_from_branch_builder(self, relpath):
    self.assertEqual(self.branch_format_from._matchingcontroldir.get_branch_format(), self.branch_format_from)
    return branchbuilder.BranchBuilder(self.get_transport(relpath), format=self.branch_format_from._matchingcontroldir)