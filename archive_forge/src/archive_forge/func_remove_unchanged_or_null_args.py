from __future__ import absolute_import, division, print_function
import datetime
import uuid
def remove_unchanged_or_null_args(server_resp, **kwargs):
    params = remove_null_args(**kwargs)
    if hasattr(server_resp, 'attrs') is False or type(server_resp.attrs) is not dict:
        return (params, params)
    params_to_search = params.copy()
    changed_attrs_dict = {}
    for key, value in params_to_search.items():
        server_value = server_resp.attrs.get(key)
        if type(server_value) is list and type(value) is dict:
            if len(value) == 0:
                continue
            temp_server_metadata_dict = {}
            for server_entry in server_value:
                temp_server_metadata_dict[server_entry['key']] = server_entry['value']
            if (value.items() <= temp_server_metadata_dict.items()) is False:
                changed_attrs_dict[key] = value
            else:
                params.pop(key)
        elif type(server_value) is dict and type(value) is dict:
            if len(value) == 0:
                continue
            if (value.items() <= server_value.items()) is False:
                changed_attrs_dict[key] = value
            else:
                params.pop(key)
        elif type(server_value) is list and type(value) is list:
            found_changed_list = False
            if len(value) != len(server_value):
                changed_attrs_dict[key] = value
                continue
            for entry_to_check in value:
                if type(entry_to_check) is dict:
                    if is_dict_item_present_on_server(server_value, entry_to_check) is True:
                        continue
                    changed_attrs_dict[key] = value
                    found_changed_list = True
                elif server_value.sort() != value.sort():
                    changed_attrs_dict[key] = value
                    found_changed_list = True
                break
            if found_changed_list is False:
                params.pop(key)
        elif server_value is None and type(value) is list:
            if len(value) == 0:
                continue
            changed_attrs_dict[key] = value
        elif server_value != value:
            if key != 'force':
                changed_attrs_dict[key] = value
        else:
            params.pop(key)
    return (changed_attrs_dict, params)