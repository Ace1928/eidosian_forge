import itertools
from typing import Callable, Dict, Tuple, Optional
from dulwich.errors import NotCommitError
from dulwich.objects import ObjectID
from dulwich.object_store import ObjectStoreGraphWalker
from dulwich.pack import PACK_SPOOL_FILE_MAX_SIZE
from dulwich.protocol import CAPABILITY_THIN_PACK, ZERO_SHA
from dulwich.refs import SYMREF
from dulwich.walk import Walker
from .. import config, trace, ui
from ..errors import (DivergedBranches, FetchLimitUnsupported,
from ..repository import FetchResult, InterRepository, AbstractSearchResult
from ..revision import NULL_REVISION, RevisionID
from .errors import NoPushSupport
from .fetch import DetermineWantsRecorder, import_git_objects
from .mapping import needs_roundtripping
from .object_store import get_object_store
from .push import MissingObjectsIterator, remote_divergence
from .refs import is_tag, ref_to_tag_name
from .remote import RemoteGitError, RemoteGitRepository
from .repository import GitRepository, GitRepositoryFormat, LocalGitRepository
from .unpeel_map import UnpeelMap
def fetch_revs(self, revs, lossy: bool, limit: Optional[int]=None) -> RevidMap:
    if not lossy and (not self.mapping.roundtripping):
        for git_sha, bzr_revid in revs:
            if bzr_revid is not None and needs_roundtripping(self.source, bzr_revid):
                raise NoPushSupport(self.source, self.target, self.mapping, bzr_revid)
    with self.source_store.lock_read():
        todo = list(self.missing_revisions(revs))[:limit]
        revidmap = {}
        with ui.ui_factory.nested_progress_bar() as pb:
            object_generator = MissingObjectsIterator(self.source_store, self.source, pb)
            for old_revid, git_sha in object_generator.import_revisions(todo, lossy=lossy):
                if lossy:
                    new_revid = self.mapping.revision_id_foreign_to_bzr(git_sha)
                else:
                    new_revid = old_revid
                    try:
                        self.mapping.revision_id_bzr_to_foreign(old_revid)
                    except InvalidRevisionId:
                        pass
                revidmap[old_revid] = (git_sha, new_revid)
            self.target_store.add_objects(object_generator)
            return revidmap