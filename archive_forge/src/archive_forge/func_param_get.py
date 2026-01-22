from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def param_get(cursor, module, name, is_guc_list_quote):
    query = 'SELECT name, setting, unit, context, boot_val FROM pg_settings WHERE name = %(name)s'
    try:
        cursor.execute(query, {'name': name})
        info = cursor.fetchone()
        cursor.execute('SHOW %s' % name)
        val = cursor.fetchone()
    except Exception as e:
        module.fail_json(msg='Unable to get %s value due to : %s' % (name, to_native(e)))
    if not info:
        module.fail_json(msg='No such parameter: %s. Please check its spelling or presence in your PostgreSQL version (https://www.postgresql.org/docs/current/runtime-config.html)' % name)
    current_val = val[name]
    raw_val = info['setting']
    unit = info['unit']
    context = info['context']
    boot_val = info['boot_val']
    if current_val == 'True':
        current_val = 'on'
    elif current_val == 'False':
        current_val = 'off'
    if unit == 'kB':
        if int(raw_val) > 0:
            raw_val = int(raw_val) * 1024
        if int(boot_val) > 0:
            boot_val = int(boot_val) * 1024
        unit = 'b'
    elif unit == 'MB':
        if int(raw_val) > 0:
            raw_val = int(raw_val) * 1024 * 1024
        if int(boot_val) > 0:
            boot_val = int(boot_val) * 1024 * 1024
        unit = 'b'
    if is_guc_list_quote:
        current_val = param_guc_list_unquote(current_val)
        raw_val = param_guc_list_unquote(raw_val)
    return {'current_val': current_val, 'raw_val': raw_val, 'unit': unit, 'boot_val': boot_val, 'context': context}