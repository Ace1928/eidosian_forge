import shutil
import sys
from breezy import (branch, controldir, errors, info, osutils, tests, upgrade,
from breezy.bzr import bzrdir
from breezy.transport import memory
def test_info_repository_hook(self):
    format = controldir.format_registry.make_controldir('knit')

    def repo_info(repo, stats, outf):
        outf.write('more info\n')
    info.hooks.install_named_hook('repository', repo_info, None)
    repo = self.make_repository('repo', shared=True, format=format)
    out, err = self.run_bzr('info -v repo')
    self.assertEqualDiff('Shared repository with trees (format: dirstate or dirstate-tags or knit)\nLocation:\n  shared repository: repo\n\nFormat:\n       control: Meta directory format 1\n    repository: {}\n\nControl directory:\n         0 branches\n\nCreate working tree for new branches inside the repository.\n\nRepository:\n         0 revisions\nmore info\n'.format(format.repository_format.get_format_description()), out)
    self.assertEqual('', err)