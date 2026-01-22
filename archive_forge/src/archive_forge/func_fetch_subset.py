from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
import re
def fetch_subset(valid_subset_list, info_subset):
    if valid_subset_list is None or isinstance(valid_subset_list, list) is False:
        return {}
    try:
        result_dict = {}
        resp = None
        for subset in valid_subset_list:
            result = {}
            try:
                if subset['name'] == 'minimum':
                    result, flag = fetch_minimum_subset(info_subset)
                    if flag is False:
                        raise Exception(result)
                elif subset['name'] == 'config':
                    result, flag = fetch_config_subset(info_subset)
                    if flag is False:
                        raise Exception(result)
                elif subset['name'] == 'all':
                    result = fetch_snapshots_for_all_subset(subset, info_subset['all'])
                    for key, value in result.items():
                        result_dict[key] = value
                    continue
                else:
                    if subset['name'] == 'user_policies' and utils.is_array_version_above_or_equal(info_subset['arrays'], '5.1.0') is False:
                        continue
                    cl_obj_set = info_subset[subset['name']]
                    query = subset['query']
                    if query is not None:
                        resp = cl_obj_set.list(detail=subset['detail'], **query, fields=subset['fields'], limit=subset['limit'])
                    else:
                        resp = cl_obj_set.list(detail=subset['detail'], fields=subset['fields'], limit=subset['limit'])
                    if resp is not None and resp.__len__() != 0:
                        if subset['count'] != -1 and resp.__len__() > subset['count']:
                            resp = resp[:subset['count']]
                        result[subset['name']] = generate_dict('data', resp)['data']
                    else:
                        result[subset['name']] = resp
                for key, value in result.items():
                    result_dict[key] = value
            except Exception as ex:
                msg = f"Failed to fetch {subset['name']} details. Error:'{str(ex)}'"
                raise Exception(msg) from ex
        return result_dict
    except Exception:
        raise