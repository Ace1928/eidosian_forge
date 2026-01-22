from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _camel_to_snake
def route_to_dict(self, route):
    return dict(name=route.name, source=_camel_to_snake(route.source), endpoint_name=route.endpoint_names[0], enabled=route.is_enabled, condition=route.condition)