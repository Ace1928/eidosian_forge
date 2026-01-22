from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
import re
def parse_subset_list(info_subset, gather_subset):
    valid_subset_list = []
    try:
        if gather_subset is None or isinstance(gather_subset, list) is False:
            add_to_valid_subset_list(valid_subset_list, 'minimum', None)
            return valid_subset_list
        for object_set in gather_subset:
            object_set_type = type(object_set)
            if object_set_type is dict:
                for key, subset_options in object_set.items():
                    key = key.strip()
                    if info_subset.get(key, None) is None:
                        raise_invalid_subset_ex(key)
                    flag, param_key, err_msg = is_subset_option_valid(subset_options)
                    if flag is False:
                        msg = f"Invalid subset option '{param_key}' provided for subset '{key}'."
                        raise Exception(msg + ' ' + err_msg)
                    else:
                        if key == 'all':
                            if is_subset_already_added('minimum', valid_subset_list) is True:
                                raise_subset_mutually_exclusive_ex()
                            handle_all_subset(info_subset, valid_subset_list, subset_options)
                            continue
                        if key == 'minimum' or key == 'config':
                            if subset_options is not None:
                                raise Exception("Subset options cannot be used with 'minimum' and 'config' subset.")
                            if key == 'minimum':
                                if is_subset_already_added('all', valid_subset_list) is True:
                                    raise_subset_mutually_exclusive_ex()
                        elif is_subset_already_added(key, valid_subset_list) is True:
                            raise_repeat_subset_ex(key)
                        add_to_valid_subset_list(valid_subset_list, key, subset_options)
            elif object_set_type is str:
                key = object_set.strip()
                if info_subset.get(key, None) is None:
                    raise_invalid_subset_ex(key)
                if is_subset_already_added(key, valid_subset_list) is True:
                    raise_repeat_subset_ex(key)
                if key == 'all':
                    if is_subset_already_added('minimum', valid_subset_list) is True:
                        raise_subset_mutually_exclusive_ex()
                    handle_all_subset(info_subset, valid_subset_list, None)
                    continue
                add_to_valid_subset_list(valid_subset_list, key, None)
        return valid_subset_list
    except Exception as ex:
        raise ex