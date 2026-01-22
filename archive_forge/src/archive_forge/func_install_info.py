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
def install_info(self):
    """
        Returns role install info
        """
    if self._install_info is None:
        info_path = os.path.join(self.path, self.META_INSTALL)
        if os.path.isfile(info_path):
            try:
                f = open(info_path, 'r')
                self._install_info = yaml_load(f)
            except Exception:
                display.vvvvv('Unable to load Galaxy install info for %s' % self.name)
                return False
            finally:
                f.close()
    return self._install_info