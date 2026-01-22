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
def rewrite_instead_of(location, purpose):
    from dulwich.config import StackedConfig, iter_instead_of
    config = StackedConfig.default()
    push = purpose != 'read'
    longest_needle = ''
    updated_url = location
    for needle, replacement in iter_instead_of(config, push):
        if not location.startswith(needle):
            continue
        if len(longest_needle) < len(needle):
            longest_needle = needle
            if longest_needle == 'lp:':
                import breezy.plugins
                if hasattr(breezy.plugins, 'launchpad'):
                    trace.warning('Ignoring insteadOf lp: in git config, because the Launchpad plugin is loaded.')
                    continue
            updated_url = replacement + location[len(needle):]
    return updated_url