import os
import sys
from .. import __version__ as breezy_version  # noqa: F401
from .. import errors as brz_errors
from .. import trace, urlutils, version_info
from ..commands import plugin_cmds
from ..controldir import ControlDirFormat, Prober, format_registry
from ..controldir import \
from ..transport import (register_lazy_transport, register_transport_proto,
from ..revisionspec import RevisionSpec_dwim, revspec_registry
from ..hooks import install_lazy_named_hook
from ..location import hooks as location_hooks
from ..repository import format_registry as repository_format_registry
from ..repository import \
from ..branch import network_format_registry as branch_network_format_registry
from ..branch import format_registry as branch_format_registry
from ..workingtree import format_registry as workingtree_format_registry
from ..diff import format_registry as diff_format_registry
from ..send import format_registry as send_format_registry
from ..directory_service import directories
from ..help_topics import topic_registry
from ..foreign import foreign_vcs_registry
from ..config import Option, bool_from_store, option_registry
def update_git_cache(repository, revid):
    """Update the git cache after a local commit."""
    if getattr(repository, '_git', None) is not None:
        return
    if not repository.control_transport.has('git'):
        return
    try:
        lazy_check_versions()
    except brz_errors.DependencyNotPresent as e:
        trace.mutter('not updating git map for %r: %s', repository, e)
    from .object_store import BazaarObjectStore
    store = BazaarObjectStore(repository)
    with store.lock_write():
        try:
            parent_revisions = set(repository.get_parent_map([revid])[revid])
        except KeyError:
            return
        missing_revisions = store._missing_revisions(parent_revisions)
        if not missing_revisions:
            store._cache.idmap.start_write_group()
            try:
                store._update_sha_map_revision(revid)
            except BaseException:
                store._cache.idmap.abort_write_group()
                raise
            else:
                store._cache.idmap.commit_write_group()