import os
import re
import unittest
from breezy import bzr, config, controldir, errors, osutils, repository, tests
from breezy.bzr.groupcompress_repo import RepositoryFormat2a
def test_repository_deprecation_warning(self):
    """Old formats give a warning"""
    self.make_obsolete_repo('foo')
    self.enable_deprecation_warning()
    out, err = self.run_bzr('status', working_dir='foo')
    self.check_warning(True)