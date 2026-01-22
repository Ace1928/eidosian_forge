from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_addon_profiles_spec():
    """
    Helper method to parse the ADDONS dictionary and generate the addon spec
    """
    spec = dict()
    for key in ADDONS.keys():
        values = ADDONS[key]
        addon_spec = dict(enabled=dict(type='bool', default=True))
        configs = values.get('config') or {}
        for item in configs.keys():
            addon_spec[item] = dict(type='str', aliases=[configs[item]], required=True)
        spec[key] = dict(type='dict', options=addon_spec, aliases=[values['name']])
    return spec