from __future__ import (absolute_import, division, print_function)
import itertools
import operator
import os
from copy import copy as shallowcopy
from functools import cache
from jinja2.exceptions import UndefinedError
from ansible import constants as C
from ansible import context
from ansible.errors import AnsibleError, AnsibleParserError, AnsibleUndefinedVariable, AnsibleAssertionError
from ansible.module_utils.six import string_types
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.common.text.converters import to_text, to_native
from ansible.parsing.dataloader import DataLoader
from ansible.playbook.attribute import Attribute, FieldAttribute, ConnectionFieldAttribute, NonInheritableFieldAttribute
from ansible.plugins.loader import module_loader, action_loader
from ansible.utils.collection_loader._collection_finder import _get_collection_metadata, AnsibleCollectionRef
from ansible.utils.display import Display
from ansible.utils.sentinel import Sentinel
from ansible.utils.vars import combine_vars, isidentifier, get_unique_id
def get_search_path(self):
    """
        Return the list of paths you should search for files, in order.
        This follows role/playbook dependency chain.
        """
    path_stack = []
    dep_chain = self.get_dep_chain()
    if dep_chain:
        path_stack.extend(reversed([x._role_path for x in dep_chain if hasattr(x, '_role_path')]))
    task_dir = os.path.dirname(self.get_path())
    if task_dir not in path_stack:
        path_stack.append(task_dir)
    return path_stack