import os
import smtplib
from breezy import gpg, merge_directive, tests, workingtree
def test_message(self):
    self.prepare_merge_directive()
    md_text = self.run_bzr('merge-directive ../tree2')[0]
    self.assertNotContainsRe(md_text, 'message: Message for merge')
    md_text = self.run_bzr('merge-directive -m Message_for_merge')[0]
    self.assertContainsRe(md_text, 'message: Message_for_merge')