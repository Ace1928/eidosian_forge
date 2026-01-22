import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_target_not_branch(self):
    """Test the formatting of TargetNotBranch."""
    error = errors.TargetNotBranch('foo')
    self.assertEqual('Your branch does not have all of the revisions required in order to merge this merge directive and the target location specified in the merge directive is not a branch: foo.', str(error))