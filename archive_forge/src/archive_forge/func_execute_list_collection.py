from __future__ import (absolute_import, division, print_function)
from ansible.cli import CLI
import argparse
import functools
import json
import os.path
import pathlib
import re
import shutil
import sys
import textwrap
import time
import typing as t
from dataclasses import dataclass
from yaml.error import YAMLError
import ansible.constants as C
from ansible import context
from ansible.cli.arguments import option_helpers as opt_help
from ansible.errors import AnsibleError, AnsibleOptionsError
from ansible.galaxy import Galaxy, get_collections_galaxy_meta_info
from ansible.galaxy.api import GalaxyAPI, GalaxyError
from ansible.galaxy.collection import (
from ansible.galaxy.collection.concrete_artifact_manager import (
from ansible.galaxy.collection.gpg import GPG_ERROR_MAP
from ansible.galaxy.dependency_resolution.dataclasses import Requirement
from ansible.galaxy.role import GalaxyRole
from ansible.galaxy.token import BasicAuthToken, GalaxyToken, KeycloakToken, NoTokenSentinel
from ansible.module_utils.ansible_release import __version__ as ansible_version
from ansible.module_utils.common.collections import is_iterable
from ansible.module_utils.common.yaml import yaml_dump, yaml_load
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils import six
from ansible.parsing.dataloader import DataLoader
from ansible.parsing.yaml.loader import AnsibleLoader
from ansible.playbook.role.requirement import RoleRequirement
from ansible.template import Templar
from ansible.utils.collection_loader import AnsibleCollectionConfig
from ansible.utils.display import Display
from ansible.utils.plugin_docs import get_versioned_doclink
@with_collection_artifacts_manager
def execute_list_collection(self, artifacts_manager=None):
    """
        List all collections installed on the local system

        :param artifacts_manager: Artifacts manager.
        """
    if artifacts_manager is not None:
        artifacts_manager.require_build_metadata = False
    output_format = context.CLIARGS['output_format']
    collection_name = context.CLIARGS['collection']
    default_collections_path = set(C.COLLECTIONS_PATHS)
    collections_search_paths = set(context.CLIARGS['collections_path'] or []) | default_collections_path | set(AnsibleCollectionConfig.collection_paths)
    collections_in_paths = {}
    warnings = []
    path_found = False
    collection_found = False
    namespace_filter = None
    collection_filter = None
    if collection_name:
        validate_collection_name(collection_name)
        namespace_filter, collection_filter = collection_name.split('.')
    collections = list(find_existing_collections(list(collections_search_paths), artifacts_manager, namespace_filter=namespace_filter, collection_filter=collection_filter, dedupe=False))
    seen = set()
    fqcn_width, version_width = _get_collection_widths(collections)
    for collection in sorted(collections, key=lambda c: c.src):
        collection_found = True
        collection_path = pathlib.Path(to_text(collection.src)).parent.parent.as_posix()
        if output_format in {'yaml', 'json'}:
            collections_in_paths.setdefault(collection_path, {})
            collections_in_paths[collection_path][collection.fqcn] = {'version': collection.ver}
        else:
            if collection_path not in seen:
                _display_header(collection_path, 'Collection', 'Version', fqcn_width, version_width)
                seen.add(collection_path)
            _display_collection(collection, fqcn_width, version_width)
    path_found = False
    for path in collections_search_paths:
        if not os.path.exists(path):
            if path in default_collections_path:
                continue
            warnings.append('- the configured path {0} does not exist.'.format(path))
        elif os.path.exists(path) and (not os.path.isdir(path)):
            warnings.append('- the configured path {0}, exists, but it is not a directory.'.format(path))
        else:
            path_found = True
    if collection_found and collection_name:
        warnings = []
    for w in warnings:
        display.warning(w)
    if not collections and (not path_found):
        raise AnsibleOptionsError('- None of the provided paths were usable. Please specify a valid path with --{0}s-path'.format(context.CLIARGS['type']))
    if output_format == 'json':
        display.display(json.dumps(collections_in_paths))
    elif output_format == 'yaml':
        display.display(yaml_dump(collections_in_paths))
    return 0