import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_bug_broken(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a', 'b'])
    tree.add('a')
    self.wt_commit(tree, 'simple log message', rev_id=b'a1', revprops={'bugs': 'test://bua g/id fixed'})
    logfile = self.make_utf8_encoded_stringio()
    formatter = log.LongLogFormatter(to_file=logfile)
    log.show_log(tree.branch, formatter)
    self.assertContainsRe(logfile.getvalue(), b"brz: ERROR: breezy.bugtracker.InvalidLineInBugsProperty: Invalid line in bugs property: 'test://bua g/id fixed'")
    text = logfile.getvalue()
    self.assertEqualDiff(text[text.index(b'-' * 60):], b'------------------------------------------------------------\nrevno: 1\ncommitter: Joe Foo <joe@foo.com>\nbranch nick: work\ntimestamp: Tue 2005-11-22 00:00:00 +0000\nmessage:\n  simple log message\n')