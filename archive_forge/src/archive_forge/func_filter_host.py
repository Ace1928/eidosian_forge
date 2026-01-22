from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils.common._collections_compat import Mapping
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import string_types
def filter_host(inventory_plugin, host, host_vars, filters):
    """
    Determine whether a host should be accepted (``True``) or not (``False``).
    """
    vars = {'inventory_hostname': host}
    if host_vars:
        vars.update(host_vars)

    def evaluate(condition):
        if isinstance(condition, bool):
            return condition
        conditional = '{%% if %s %%} True {%% else %%} False {%% endif %%}' % condition
        templar = inventory_plugin.templar
        old_vars = templar.available_variables
        try:
            templar.available_variables = vars
            return boolean(templar.template(conditional))
        except Exception as e:
            raise AnsibleParserError('Could not evaluate filter condition {condition!r} for host {host}: {err}'.format(host=host, condition=condition, err=to_native(e)))
        finally:
            templar.available_variables = old_vars
    for filter in filters:
        if 'include' in filter:
            expr = filter['include']
            if evaluate(expr):
                return True
        if 'exclude' in filter:
            expr = filter['exclude']
            if evaluate(expr):
                return False
    return True