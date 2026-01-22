from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from copy import deepcopy
from ansible.module_utils.common.dict_transformations import (
def compare_dicts(old_params, new_params, param_name):
    """Compare two dictionaries using recursive_diff method and assuming that null values coming from yaml input
    are acting like absent values"""
    oldd = old_params.get(param_name, {})
    newd = new_params.get(param_name, {})
    if oldd == {} and newd == {}:
        return True
    diffs = recursive_diff(oldd, newd)
    if diffs is None:
        return True
    else:
        actual_diffs = diffs[1]
        return all((value is None or not value for value in actual_diffs.values()))