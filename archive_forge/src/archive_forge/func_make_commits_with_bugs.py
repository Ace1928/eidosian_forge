import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def make_commits_with_bugs(self):
    """Helper method for LogFormatter tests"""
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a', 'b'])
    tree.add('a')
    self.wt_commit(tree, 'simple log message', rev_id=b'a1', revprops={'bugs': 'test://bug/id fixed'})
    tree.add('b')
    self.wt_commit(tree, 'multiline\nlog\nmessage\n', rev_id=b'a2', authors=['Joe Bar <joe@bar.com>'], revprops={'bugs': 'test://bug/id fixed\ntest://bug/2 fixed'})
    return tree