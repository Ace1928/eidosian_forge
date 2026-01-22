import os
import re
import unittest
from breezy import bzr, config, controldir, errors, osutils, repository, tests
from breezy.bzr.groupcompress_repo import RepositoryFormat2a
def test_repository_deprecation_warning_suppressed_locations(self):
    """Old formats give a warning"""
    self.make_obsolete_repo('foo')
    conf = config.LocationStack(osutils.pathjoin(self.test_dir, 'foo'))
    conf.set('suppress_warnings', 'format_deprecation')
    self.enable_deprecation_warning()
    out, err = self.run_bzr('status', working_dir='foo')
    self.check_warning(False)