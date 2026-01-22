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
def lookup_bzr_revision_id(self, bzr_revid, mapping=None):
    """Lookup a bzr revision id in a Git repository.

        :param bzr_revid: Bazaar revision id
        :param mapping: Optional mapping to use
        :return: Tuple with git commit id, mapping that was used and supplement
            details
        """
    try:
        git_sha, mapping = mapping_registry.revision_id_bzr_to_foreign(bzr_revid)
    except errors.InvalidRevisionId:
        raise errors.NoSuchRevision(self, bzr_revid)
    else:
        return (git_sha, mapping)