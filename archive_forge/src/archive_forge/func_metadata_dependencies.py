from __future__ import (absolute_import, division, print_function)
import errno
import datetime
import functools
import os
import tarfile
import tempfile
from collections.abc import MutableSequence
from shutil import rmtree
from ansible import context
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.galaxy.api import GalaxyAPI
from ansible.galaxy.user_agent import user_agent
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.yaml import yaml_dump, yaml_load
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.urls import open_url
from ansible.playbook.role.requirement import RoleRequirement
from ansible.utils.display import Display
from ansible.utils.path import is_subpath, unfrackpath
@property
def metadata_dependencies(self):
    """
        Returns a list of dependencies from role metadata
        """
    if self._metadata_dependencies is None:
        self._metadata_dependencies = []
        if self.metadata is not None:
            self._metadata_dependencies = self.metadata.get('dependencies') or []
    if not isinstance(self._metadata_dependencies, MutableSequence):
        raise AnsibleParserError(f'Expected role dependencies to be a list. Role {self} has meta/main.yml with dependencies {self._metadata_dependencies}')
    return self._metadata_dependencies