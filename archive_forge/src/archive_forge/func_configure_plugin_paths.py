from __future__ import annotations
import json
import os
import shutil
import typing as t
from .constants import (
from .io import (
from .util import (
from .util_common import (
from .config import (
from .data import (
from .python_requirements import (
from .host_configs import (
from .thread import (
def configure_plugin_paths(args: CommonConfig) -> dict[str, str]:
    """Return environment variables with paths to plugins relevant for the current command."""
    if not isinstance(args, IntegrationConfig):
        return {}
    support_path = os.path.join(ANSIBLE_SOURCE_ROOT, 'test', 'support', args.command)
    collection_root = os.path.join(support_path, 'collections')
    env = dict(ANSIBLE_COLLECTIONS_PATH=collection_root)
    plugin_root = os.path.join(support_path, 'plugins')
    plugin_list = ['action', 'become', 'cache', 'callback', 'cliconf', 'connection', 'filter', 'httpapi', 'inventory', 'lookup', 'netconf', 'strategy', 'terminal', 'test', 'vars']
    plugin_map = dict((('%s_plugins' % name, name) for name in plugin_list))
    plugin_map.update(doc_fragment='doc_fragments', library='modules', module_utils='module_utils')
    env.update(dict((('ANSIBLE_%s' % key.upper(), os.path.join(plugin_root, value)) for key, value in plugin_map.items())))
    env = dict(((key, value) for key, value in env.items() if os.path.isdir(value)))
    return env