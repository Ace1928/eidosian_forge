from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.utils.plugins.module_utils.common.utils import dict_merge
def check_argspec(schema, name, schema_format='doc', schema_conditionals=None, **args):
    if schema_conditionals is None:
        schema_conditionals = {}
    aav = AnsibleArgSpecValidator(data=args, schema=schema, schema_format=schema_format, schema_conditionals=schema_conditionals, name=name)
    result = {}
    valid, errors, updated_params = aav.validate()
    if not valid:
        result['errors'] = errors
        result['failed'] = True
        result['msg'] = 'argspec validation failed for {name} plugin'.format(name=name)
    return (valid, result, updated_params)