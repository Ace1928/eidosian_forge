from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
def deep_created_origin_to_dict(origin):
    return dict(name=origin.name, host_name=origin.host_name, http_port=origin.http_port, https_port=origin.https_port)