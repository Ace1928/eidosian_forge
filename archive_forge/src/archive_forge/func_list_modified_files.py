import re
import shutil
import tempfile
from typing import Any, List, Optional
from ... import branch as _mod_branch
from ... import controldir, errors, hooks, urlutils
from ...forge import (AutoMergeUnsupported, Forge, LabelsUnsupported,
from ...git.urls import git_url_to_bzr_url
from ...lazy_import import lazy_import
from ...trace import mutter
from breezy.plugins.launchpad import (
from ...transport import get_transport
def list_modified_files():
    lca_tree = self.source_branch_lp.find_lca_tree(self.target_branch_lp)
    source_tree = self.source_branch.basis_tree()
    files = modified_files(lca_tree, source_tree)
    return list(files)