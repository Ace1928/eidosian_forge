from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError
from ansible.module_utils.six import string_types
from ansible.playbook.role.definition import RoleDefinition
from ansible.utils.display import Display
from ansible.utils.galaxy import scm_archive_resource
@staticmethod
def repo_url_to_role_name(repo_url):
    if '://' not in repo_url and '@' not in repo_url:
        return repo_url
    trailing_path = repo_url.split('/')[-1]
    if trailing_path.endswith('.git'):
        trailing_path = trailing_path[:-4]
    if trailing_path.endswith('.tar.gz'):
        trailing_path = trailing_path[:-7]
    if ',' in trailing_path:
        trailing_path = trailing_path.split(',')[0]
    return trailing_path