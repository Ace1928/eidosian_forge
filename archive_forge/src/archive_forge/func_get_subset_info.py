from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
import re
def get_subset_info(client_obj, gather_subset):
    if utils.is_null_or_empty(gather_subset):
        return (False, False, 'Please provide atleast one subset.', {})
    result_dict = []
    try:
        info_subset = intialize_info_subset(client_obj)
        valid_subset_list = parse_subset_list(info_subset, gather_subset)
        if valid_subset_list is not None and valid_subset_list.__len__() > 0:
            result_dict = fetch_subset(valid_subset_list, info_subset)
            return (True, False, 'Fetched the subset details.', result_dict)
        else:
            return (True, False, 'No vaild subset provided.', result_dict)
    except Exception as ex:
        return (False, False, f'{ex}', {})