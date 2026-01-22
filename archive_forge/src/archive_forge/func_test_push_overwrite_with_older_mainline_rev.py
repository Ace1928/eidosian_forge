import os
from io import BytesIO
from ... import (branch, builtins, check, controldir, errors, push, revision,
from ...bzr import branch as bzrbranch
from ...bzr.smart import client
from .. import per_branch, test_server
def test_push_overwrite_with_older_mainline_rev(self):
    """Pushing an older mainline revision with overwrite.

        This was <https://bugs.launchpad.net/bzr/+bug/386576>.
        """
    source = self.make_branch_and_tree('source')
    target = self.make_branch('target')
    source.commit('1st commit')
    rev2 = source.commit('2nd commit')
    source.commit('3rd commit')
    source.branch.push(target)
    source.branch.push(target, stop_revision=rev2, overwrite=True)
    self.assertEqual(rev2, target.last_revision())