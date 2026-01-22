import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_show_branch_change_no_change(self):
    tree = self.setup_ab_tree()
    s = StringIO()
    log.show_branch_change(tree.branch, s, 3, b'3b')
    self.assertEqual(s.getvalue(), 'Nothing seems to have changed\n')