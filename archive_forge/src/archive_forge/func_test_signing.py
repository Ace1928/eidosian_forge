import os
import smtplib
from breezy import gpg, merge_directive, tests, workingtree
def test_signing(self):
    self.prepare_merge_directive()
    old_strategy = gpg.GPGStrategy
    gpg.GPGStrategy = gpg.LoopbackGPGStrategy
    try:
        md_text = self.run_bzr('merge-directive --sign ../tree2')[0]
    finally:
        gpg.GPGStrategy = old_strategy
    self.assertContainsRe(md_text, '^-----BEGIN PSEUDO-SIGNED CONTENT')