import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def setup_ab_tree(self):
    tree = self.setup_a_tree()
    tree.set_last_revision(b'1a')
    tree.branch.set_last_revision_info(1, b'1a')
    kwargs = {'committer': 'Joe Foo <joe@foo.com>', 'timestamp': 1132617600, 'timezone': 0}
    tree.commit('commit 2b', rev_id=b'2b', **kwargs)
    tree.commit('commit 3b', rev_id=b'3b', **kwargs)
    return tree