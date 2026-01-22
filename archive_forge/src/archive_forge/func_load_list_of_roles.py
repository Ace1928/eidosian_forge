from __future__ import (absolute_import, division, print_function)
import os
from ansible import constants as C
from ansible.errors import AnsibleParserError, AnsibleUndefinedVariable, AnsibleAssertionError
from ansible.module_utils.common.text.converters import to_native
from ansible.parsing.mod_args import ModuleArgsParser
from ansible.utils.display import Display
def load_list_of_roles(ds, play, current_role_path=None, variable_manager=None, loader=None, collection_search_list=None):
    """
    Loads and returns a list of RoleInclude objects from the ds list of role definitions
    :param ds: list of roles to load
    :param play: calling Play object
    :param current_role_path: path of the owning role, if any
    :param variable_manager: varmgr to use for templating
    :param loader: loader to use for DS parsing/services
    :param collection_search_list: list of collections to search for unqualified role names
    :return:
    """
    from ansible.playbook.role.include import RoleInclude
    if not isinstance(ds, list):
        raise AnsibleAssertionError('ds (%s) should be a list but was a %s' % (ds, type(ds)))
    roles = []
    for role_def in ds:
        i = RoleInclude.load(role_def, play=play, current_role_path=current_role_path, variable_manager=variable_manager, loader=loader, collection_list=collection_search_list)
        roles.append(i)
    return roles