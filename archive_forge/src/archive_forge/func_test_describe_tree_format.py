import sys
from .. import branch as _mod_branch
from .. import controldir, errors, info
from .. import repository as _mod_repository
from .. import tests, workingtree
from ..bzr import branch as _mod_bzrbranch
def test_describe_tree_format(self):
    for key, format in controldir.format_registry.iteritems():
        if key in controldir.format_registry.aliases():
            continue
        if not format().supports_workingtrees:
            continue
        self.assertTreeDescription(key)