import gzip
import os
import re
import tempfile
from .... import tests
from ....tests import features
from ....tests.blackbox import ExternalBase
from ..cmds import _get_source_stream
from . import FastimportFeature
from :1
from :2
from :1
from :2
def test_missing_bytes(self):
    self.build_tree_contents([('empty.fi', b'\ncommit refs/heads/master\nmark :1\ncommitter\ndata 15\n')])
    self.make_branch_and_tree('br')
    self.run_bzr_error(['brz: ERROR: 4: Parse error: line 4: Command .*commit.* is missing section .*committer.*\n'], 'fast-import empty.fi br')