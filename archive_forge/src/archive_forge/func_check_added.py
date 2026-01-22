import os
from breezy.branch import Branch
from breezy.tests import TestCaseWithTransport
def check_added(expected, null=False):
    command = 'added'
    if null:
        command += ' --null'
    out, err = self.run_bzr(command)
    self.assertEqual(out, expected)
    self.assertEqual(err, '')