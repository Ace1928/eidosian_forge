from __future__ import (absolute_import, division, print_function)
from ansible import constants as C
from ansible import context
from ansible.errors import AnsibleParserError, AnsibleAssertionError
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.six import binary_type, string_types, text_type
from ansible.playbook.attribute import NonInheritableFieldAttribute
from ansible.playbook.base import Base
from ansible.playbook.block import Block
from ansible.playbook.collectionsearch import CollectionSearch
from ansible.playbook.helpers import load_list_of_blocks, load_list_of_roles
from ansible.playbook.role import Role, hash_params
from ansible.playbook.task import Task
from ansible.playbook.taggable import Taggable
from ansible.vars.manager import preprocess_vars
from ansible.utils.display import Display
def compile_roles_handlers(self):
    """
        Handles the role handler compilation step, returning a flat list of Handlers
        This is done for all roles in the Play.
        """
    block_list = []
    if len(self.roles) > 0:
        for r in self.roles:
            if r.from_include:
                continue
            block_list.extend(r.get_handler_blocks(play=self))
    return block_list