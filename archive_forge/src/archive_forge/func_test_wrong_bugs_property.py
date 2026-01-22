import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_wrong_bugs_property(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['foo'])
    self.wt_commit(tree, 'simple log message', rev_id=b'a1', revprops={'bugs': 'test://bug/id invalid_value'})
    logfile = self.make_utf8_encoded_stringio()
    formatter = log.ShortLogFormatter(to_file=logfile)
    log.show_log(tree.branch, formatter)
    lines = logfile.getvalue().splitlines()
    self.assertEqual(lines[0], b'    1 Joe Foo\t2005-11-22')
    self.assertEqual(lines[1], b"brz: ERROR: breezy.bugtracker.InvalidBugStatus: Invalid bug status: 'invalid_value'")
    self.assertEqual(lines[-2], b'      simple log message')