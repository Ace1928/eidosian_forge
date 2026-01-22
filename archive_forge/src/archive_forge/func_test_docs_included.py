import os
import subprocess
import sys
import unittest
@unittest.skipIf('CI' not in os.environ, 'Docs not required for local builds')
def test_docs_included(self):
    from pygame.docs.__main__ import has_local_docs
    self.assertTrue(has_local_docs())