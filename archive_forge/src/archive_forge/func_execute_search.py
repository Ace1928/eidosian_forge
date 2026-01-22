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
def execute_search(self):
    """ searches for roles on the Ansible Galaxy server"""
    page_size = 1000
    search = None
    if context.CLIARGS['args']:
        search = '+'.join(context.CLIARGS['args'])
    if not search and (not context.CLIARGS['platforms']) and (not context.CLIARGS['galaxy_tags']) and (not context.CLIARGS['author']):
        raise AnsibleError('Invalid query. At least one search term, platform, galaxy tag or author must be provided.')
    response = self.api.search_roles(search, platforms=context.CLIARGS['platforms'], tags=context.CLIARGS['galaxy_tags'], author=context.CLIARGS['author'], page_size=page_size)
    if response['count'] == 0:
        display.warning('No roles match your search.')
        return 0
    data = [u'']
    if response['count'] > page_size:
        data.append(u'Found %d roles matching your search. Showing first %s.' % (response['count'], page_size))
    else:
        data.append(u'Found %d roles matching your search:' % response['count'])
    max_len = []
    for role in response['results']:
        max_len.append(len(role['username'] + '.' + role['name']))
    name_len = max(max_len)
    format_str = u' %%-%ds %%s' % name_len
    data.append(u'')
    data.append(format_str % (u'Name', u'Description'))
    data.append(format_str % (u'----', u'-----------'))
    for role in response['results']:
        data.append(format_str % (u'%s.%s' % (role['username'], role['name']), role['description']))
    data = u'\n'.join(data)
    self.pager(data)
    return 0