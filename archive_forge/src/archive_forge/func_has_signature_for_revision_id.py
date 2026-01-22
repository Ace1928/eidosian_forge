from io import BytesIO
from dulwich.errors import NotCommitError
from dulwich.object_store import peel_sha, tree_lookup_path
from dulwich.objects import ZERO_SHA, Commit
from .. import check, errors
from .. import graph as _mod_graph
from .. import lock, repository
from .. import revision as _mod_revision
from .. import trace, transactions, ui
from ..decorators import only_raises
from ..foreign import ForeignRepository
from .filegraph import GitFileLastChangeScanner, GitFileParentProvider
from .mapping import (default_mapping, encode_git_path, foreign_vcs_git,
from .tree import GitRevisionTree
def has_signature_for_revision_id(self, revision_id):
    """Check whether a GPG signature is present for this revision.

        This is never the case for Git repositories.
        """
    try:
        self.get_signature_text(revision_id)
    except errors.NoSuchRevision:
        return False
    else:
        return True