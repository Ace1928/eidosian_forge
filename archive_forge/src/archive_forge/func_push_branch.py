import gzip
import re
from dulwich.refs import SymrefLoop
from .. import config, debug, errors, osutils, trace, ui, urlutils
from ..controldir import BranchReferenceLoop
from ..errors import (AlreadyBranchError, BzrError, ConnectionReset,
from ..push import PushResult
from ..revision import NULL_REVISION
from ..revisiontree import RevisionTree
from ..transport import (NoSuchFile, Transport,
from . import is_github_url, lazy_check_versions, user_agent_for_github
import os
import select
import urllib.parse as urlparse
import dulwich
import dulwich.client
from dulwich.errors import GitProtocolError, HangupException
from dulwich.pack import (PACK_SPOOL_FILE_MAX_SIZE, Pack, load_pack_index,
from dulwich.protocol import ZERO_SHA
from dulwich.refs import SYMREF, DictRefsContainer
from dulwich.repo import NotGitRepository
from .branch import (GitBranch, GitBranchFormat, GitBranchPushResult, GitTags,
from .dir import GitControlDirFormat, GitDir
from .errors import GitSmartRemoteNotSupported
from .mapping import encode_git_path, mapping_registry
from .object_store import get_object_store
from .push import remote_divergence
from .refs import (branch_name_to_ref, is_peeled, ref_to_tag_name,
from .repository import GitRepository, GitRepositoryFormat
def push_branch(self, source, revision_id=None, overwrite=False, remember=False, create_prefix=False, lossy=False, name=None, tag_selector=None):
    """Push the source branch into this ControlDir."""
    if revision_id is None:
        revision_id = source.last_revision()
    elif not source.repository.has_revision(revision_id):
        raise NoSuchRevision(source, revision_id)
    push_result = GitPushResult()
    push_result.workingtree_updated = None
    push_result.master_branch = None
    push_result.source_branch = source
    push_result.stacked_on = None
    push_result.branch_push_result = None
    repo = self.find_repository()
    refname = self._get_selected_ref(name)
    try:
        ref_chain, old_sha = self.get_refs_container().follow(refname)
    except NotBranchError:
        actual_refname = refname
        old_sha = None
    else:
        if ref_chain:
            actual_refname = ref_chain[-1]
        else:
            actual_refname = refname
    if isinstance(source, GitBranch) and lossy:
        raise errors.LossyPushToSameVCS(source.controldir, self)
    source_store = get_object_store(source.repository)
    fetch_tags = source.get_config_stack().get('branch.fetch_tags')

    def get_changed_refs(remote_refs):
        if self._refs is not None:
            update_refs_container(self._refs, remote_refs)
        ret = {}
        push_result.new_original_revid = revision_id
        if lossy:
            new_sha = source_store._lookup_revision_sha1(revision_id)
        else:
            try:
                new_sha = repo.lookup_bzr_revision_id(revision_id)[0]
            except errors.NoSuchRevision:
                raise errors.NoRoundtrippingSupport(source, self.open_branch(name=name, nascent_ok=True))
        old_sha = remote_refs.get(actual_refname)
        if not overwrite:
            if remote_divergence(old_sha, new_sha, source_store):
                raise DivergedBranches(source, self.open_branch(name, nascent_ok=True))
        ret[actual_refname] = new_sha
        if fetch_tags:
            for tagname, revid in source.tags.get_tag_dict().items():
                if tag_selector and (not tag_selector(tagname)):
                    continue
                if lossy:
                    try:
                        new_sha = source_store._lookup_revision_sha1(revid)
                    except KeyError:
                        if source.repository.has_revision(revid):
                            raise
                else:
                    try:
                        new_sha = repo.lookup_bzr_revision_id(revid)[0]
                    except errors.NoSuchRevision:
                        continue
                    else:
                        if not source.repository.has_revision(revid):
                            continue
                ret[tag_name_to_ref(tagname)] = new_sha
        return ret
    with source_store.lock_read():

        def generate_pack_data(have, want, progress=None, ofs_delta=True):
            git_repo = getattr(source.repository, '_git', None)
            if git_repo:
                shallow = git_repo.get_shallow()
            else:
                shallow = None
            if lossy:
                return source_store.generate_lossy_pack_data(have, want, shallow=shallow, progress=progress, ofs_delta=ofs_delta)
            elif shallow:
                return source_store.generate_pack_data(have, want, shallow=shallow, progress=progress, ofs_delta=ofs_delta)
            else:
                return source_store.generate_pack_data(have, want, progress=progress, ofs_delta=ofs_delta)
        dw_result = self.send_pack(get_changed_refs, generate_pack_data)
        new_refs = dw_result.refs
        error = dw_result.ref_status.get(actual_refname)
        if error:
            raise error
        for ref, error in dw_result.ref_status.items():
            if error:
                trace.warning('unable to open ref %s: %s', ref, error)
    push_result.new_revid = repo.lookup_foreign_revision_id(new_refs[actual_refname])
    if old_sha is not None:
        push_result.old_revid = repo.lookup_foreign_revision_id(old_sha)
    else:
        push_result.old_revid = NULL_REVISION
    if self._refs is not None:
        update_refs_container(self._refs, new_refs)
    push_result.target_branch = self.open_branch(name)
    if old_sha is not None:
        push_result.branch_push_result = GitBranchPushResult()
        push_result.branch_push_result.source_branch = source
        push_result.branch_push_result.target_branch = push_result.target_branch
        push_result.branch_push_result.local_branch = None
        push_result.branch_push_result.master_branch = push_result.target_branch
        push_result.branch_push_result.old_revid = push_result.old_revid
        push_result.branch_push_result.new_revid = push_result.new_revid
        push_result.branch_push_result.new_original_revid = push_result.new_original_revid
    if source.get_push_location() is None or remember:
        source.set_push_location(push_result.target_branch.base)
    return push_result