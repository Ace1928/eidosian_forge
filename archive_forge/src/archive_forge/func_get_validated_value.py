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
def get_validated_value(self, name, attribute, value, templar):
    if attribute.isa == 'string':
        value = to_text(value)
    elif attribute.isa == 'int':
        value = int(value)
    elif attribute.isa == 'float':
        value = float(value)
    elif attribute.isa == 'bool':
        value = boolean(value, strict=True)
    elif attribute.isa == 'percent':
        if isinstance(value, string_types) and '%' in value:
            value = value.replace('%', '')
        value = float(value)
    elif attribute.isa == 'list':
        if value is None:
            value = []
        elif not isinstance(value, list):
            value = [value]
        if attribute.listof is not None:
            for item in value:
                if not isinstance(item, attribute.listof):
                    raise AnsibleParserError("the field '%s' should be a list of %s, but the item '%s' is a %s" % (name, attribute.listof, item, type(item)), obj=self.get_ds())
                elif attribute.required and attribute.listof == string_types:
                    if item is None or item.strip() == '':
                        raise AnsibleParserError("the field '%s' is required, and cannot have empty values" % (name,), obj=self.get_ds())
    elif attribute.isa == 'set':
        if value is None:
            value = set()
        elif not isinstance(value, (list, set)):
            if isinstance(value, string_types):
                value = value.split(',')
            else:
                value = [value]
        if not isinstance(value, set):
            value = set(value)
    elif attribute.isa == 'dict':
        if value is None:
            value = dict()
        elif not isinstance(value, dict):
            raise TypeError('%s is not a dictionary' % value)
    elif attribute.isa == 'class':
        if not isinstance(value, attribute.class_type):
            raise TypeError('%s is not a valid %s (got a %s instead)' % (name, attribute.class_type, type(value)))
        value.post_validate(templar=templar)
    else:
        raise AnsibleAssertionError(f'Unknown value for attribute.isa: {attribute.isa}')
    return value