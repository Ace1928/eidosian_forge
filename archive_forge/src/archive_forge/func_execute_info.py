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
def execute_info(self):
    """
        prints out detailed information about an installed role as well as info available from the galaxy API.
        """
    roles_path = context.CLIARGS['roles_path']
    data = ''
    for role in context.CLIARGS['args']:
        role_info = {'path': roles_path}
        gr = GalaxyRole(self.galaxy, self.lazy_role_api, role)
        install_info = gr.install_info
        if install_info:
            if 'version' in install_info:
                install_info['installed_version'] = install_info['version']
                del install_info['version']
            role_info.update(install_info)
        if not context.CLIARGS['offline']:
            remote_data = None
            try:
                remote_data = self.api.lookup_role_by_name(role, False)
            except GalaxyError as e:
                if e.http_code == 400 and 'Bad Request' in e.message:
                    data = u'- the role %s was not found' % role
                    break
                raise AnsibleError("Unable to find info about '%s': %s" % (role, e))
            if remote_data:
                role_info.update(remote_data)
            else:
                data = u'- the role %s was not found' % role
                break
        elif context.CLIARGS['offline'] and (not gr._exists):
            data = u'- the role %s was not found' % role
            break
        if gr.metadata:
            role_info.update(gr.metadata)
        req = RoleRequirement()
        role_spec = req.role_yaml_parse({'role': role})
        if role_spec:
            role_info.update(role_spec)
        data += self._display_role_info(role_info)
    self.pager(data)