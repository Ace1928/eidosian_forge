from __future__ import (absolute_import, division, print_function)
import os
from collections.abc import Container, Mapping, Set, Sequence
from types import MappingProxyType
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleParserError, AnsibleAssertionError
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six import binary_type, text_type
from ansible.playbook.attribute import FieldAttribute
from ansible.playbook.base import Base
from ansible.playbook.collectionsearch import CollectionSearch
from ansible.playbook.conditional import Conditional
from ansible.playbook.delegatable import Delegatable
from ansible.playbook.helpers import load_list_of_blocks
from ansible.playbook.role.metadata import RoleMetadata
from ansible.playbook.taggable import Taggable
from ansible.plugins.loader import add_all_plugin_dirs
from ansible.utils.collection_loader import AnsibleCollectionConfig
from ansible.utils.path import is_subpath
from ansible.utils.sentinel import Sentinel
from ansible.utils.vars import combine_vars
def get_handler_blocks(self, play, dep_chain=None):
    if self._compiled_handler_blocks:
        return self._compiled_handler_blocks
    self._compiled_handler_blocks = block_list = []
    if dep_chain is None:
        dep_chain = []
    new_dep_chain = dep_chain + [self]
    for dep in self.get_direct_dependencies():
        dep_blocks = dep.get_handler_blocks(play=play, dep_chain=new_dep_chain)
        block_list.extend(dep_blocks)
    for task_block in self._handler_blocks:
        new_task_block = task_block.copy()
        new_task_block._dep_chain = new_dep_chain
        new_task_block._play = play
        block_list.append(new_task_block)
    return block_list