import os
import re
from breezy import tests, workingtree
from breezy.diff import DiffTree
from breezy.diff import format_registry as diff_format_registry
from breezy.tests import features
def check_no_diffs(self, cmd):
    out, err = self.run_bzr(cmd, retcode=0)
    self.assertEqual('', err)
    self.assertEqual('', out)