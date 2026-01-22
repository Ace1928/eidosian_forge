import os
from io import BytesIO
from ... import (branch, builtins, check, controldir, errors, push, revision,
from ...bzr import branch as bzrbranch
from ...bzr.smart import client
from .. import per_branch, test_server
def test_empty_branch_command(self):
    """The 'bzr push' command should make a limited number of HPSS calls.
        """
    cmd = builtins.cmd_push()
    cmd.outf = BytesIO()
    cmd.run(directory=self.get_url('empty'), location=self.smart_server.get_url() + 'target')
    self.assertTrue(len(self.hpss_calls) <= 9, self.hpss_calls)