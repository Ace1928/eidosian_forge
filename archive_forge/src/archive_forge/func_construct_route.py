from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel, _camel_to_snake
import re
def construct_route(self, route):
    if not route:
        return None
    return self.IoThub_models.RouteProperties(name=route['name'], source=_snake_to_camel(snake=route['source'], capitalize_first=True), is_enabled=route['enabled'], endpoint_names=[route['endpoint_name']], condition=route.get('condition'))