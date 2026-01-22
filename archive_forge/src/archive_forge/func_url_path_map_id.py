from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from copy import deepcopy
from ansible.module_utils.common.dict_transformations import (
def url_path_map_id(subscription_id, resource_group_name, appgateway_name, name):
    """Generate the id for a url path map"""
    return '/subscriptions/{0}/resourceGroups/{1}/providers/Microsoft.Network/applicationGateways/{2}/urlPathMaps/{3}'.format(subscription_id, resource_group_name, appgateway_name, name)