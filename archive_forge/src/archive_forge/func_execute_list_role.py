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
def execute_list_role(self):
    """
        List all roles installed on the local system or a specific role
        """
    path_found = False
    role_found = False
    warnings = []
    roles_search_paths = context.CLIARGS['roles_path']
    role_name = context.CLIARGS['role']
    for path in roles_search_paths:
        role_path = GalaxyCLI._resolve_path(path)
        if os.path.isdir(path):
            path_found = True
        else:
            warnings.append('- the configured path {0} does not exist.'.format(path))
            continue
        if role_name:
            gr = GalaxyRole(self.galaxy, self.lazy_role_api, role_name, path=os.path.join(role_path, role_name))
            if os.path.isdir(gr.path):
                role_found = True
                display.display('# %s' % os.path.dirname(gr.path))
                _display_role(gr)
                break
            warnings.append('- the role %s was not found' % role_name)
        else:
            if not os.path.exists(role_path):
                warnings.append('- the configured path %s does not exist.' % role_path)
                continue
            if not os.path.isdir(role_path):
                warnings.append('- the configured path %s, exists, but it is not a directory.' % role_path)
                continue
            display.display('# %s' % role_path)
            path_files = os.listdir(role_path)
            for path_file in path_files:
                gr = GalaxyRole(self.galaxy, self.lazy_role_api, path_file, path=path)
                if gr.metadata:
                    _display_role(gr)
    if role_found and role_name:
        warnings = []
    for w in warnings:
        display.warning(w)
    if not path_found:
        raise AnsibleOptionsError('- None of the provided paths were usable. Please specify a valid path with --{0}s-path'.format(context.CLIARGS['type']))
    return 0