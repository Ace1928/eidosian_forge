import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_merge_sorted_simple_revnos_exclude_ancestry(self):
    b = self.make_branch_with_alternate_ancestries()
    self.assertLogRevnos([b'3', b'2'], b, b'1', b'3', exclude_common_ancestry=True, generate_merge_revisions=False)
    self.assertLogRevnos([b'3', b'1.1.2', b'1.2.1', b'1.1.1', b'2'], b, b'1', b'3', exclude_common_ancestry=True, generate_merge_revisions=True)