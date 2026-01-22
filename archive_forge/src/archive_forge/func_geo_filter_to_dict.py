from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
def geo_filter_to_dict(geo_filter):
    return dict(relative_path=geo_filter.relative_path, action=geo_filter.action, country_codes=geo_filter.country_codes)