from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError
from ansible.module_utils.six import string_types
from ansible.playbook.role.definition import RoleDefinition
from ansible.utils.display import Display
from ansible.utils.galaxy import scm_archive_resource
@staticmethod
def scm_archive_role(src, scm='git', name=None, version='HEAD', keep_scm_meta=False):
    return scm_archive_resource(src, scm=scm, name=name, version=version, keep_scm_meta=keep_scm_meta)