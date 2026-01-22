from __future__ import (absolute_import, division, print_function)
from collections import defaultdict
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.routeros.plugins.module_utils.api import (
from ansible_collections.community.routeros.plugins.module_utils._api_data import (
def find_modifications(old_entry, new_entry, path_info, module, for_text='', return_none_instead_of_fail=False):
    modifications = OrderedDict()
    updated_entry = old_entry.copy()
    for k, v in new_entry.items():
        if k == '.id':
            continue
        disabled_k = None
        if k.startswith('!'):
            disabled_k = k[1:]
        elif v is None or v == path_info.fields[k].remove_value:
            disabled_k = k
        if disabled_k is not None:
            if disabled_k in old_entry:
                if path_info.fields[disabled_k].remove_value is not None:
                    modifications[disabled_k] = path_info.fields[disabled_k].remove_value
                else:
                    modifications['!%s' % disabled_k] = ''
                del updated_entry[disabled_k]
            continue
        if k not in old_entry and path_info.fields[k].default == v and (not path_info.fields[k].can_disable):
            continue
        key_info = path_info.fields[k]
        if key_info.read_only:
            if old_entry.get(k) != v:
                module.fail_json(msg='Read-only key "{key}" has value "{old_value}", but should have new value "{new_value}"{for_text}.'.format(key=k, old_value=old_entry.get(k), new_value=v, for_text=for_text))
            continue
        if key_info.write_only:
            if module.params['handle_write_only'] == 'create_only':
                continue
        if k not in old_entry or old_entry[k] != v:
            modifications[k] = v
            updated_entry[k] = v
    handle_entries_content = module.params['handle_entries_content']
    if handle_entries_content != 'ignore':
        for k in old_entry:
            if k == '.id' or k in new_entry or '!%s' % k in new_entry or (k not in path_info.fields):
                continue
            field_info = path_info.fields[k]
            if field_info.default is not None and field_info.default == old_entry[k]:
                continue
            if field_info.remove_value is not None and field_info.remove_value == old_entry[k]:
                continue
            if field_info.can_disable:
                if field_info.default is not None:
                    modifications[k] = field_info.default
                elif field_info.remove_value is not None:
                    modifications[k] = field_info.remove_value
                else:
                    modifications['!%s' % k] = ''
                del updated_entry[k]
            elif field_info.default is not None:
                modifications[k] = field_info.default
                updated_entry[k] = field_info.default
            elif handle_entries_content == 'remove':
                if return_none_instead_of_fail:
                    return (None, None)
                module.fail_json(msg='Key "{key}" cannot be removed{for_text}.'.format(key=k, for_text=for_text))
        for k in path_info.fields:
            field_info = path_info.fields[k]
            if k not in old_entry and k not in new_entry and field_info.can_disable and (field_info.default is not None):
                modifications[k] = field_info.default
                updated_entry[k] = field_info.default
    return (modifications, updated_entry)