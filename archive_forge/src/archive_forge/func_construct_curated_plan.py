from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def construct_curated_plan(self, plan):
    plan_facts = self.serialize_obj(plan, AZURE_OBJECT_CLASS)
    curated_output = dict()
    curated_output['id'] = plan_facts['id']
    curated_output['name'] = plan_facts['name']
    curated_output['resource_group'] = plan_facts['resource_group']
    curated_output['location'] = plan_facts['location']
    curated_output['tags'] = plan_facts.get('tags', None)
    curated_output['is_linux'] = False
    curated_output['kind'] = plan_facts['kind']
    curated_output['sku'] = plan_facts['sku']
    if plan_facts.get('reserved', None):
        curated_output['is_linux'] = True
    return curated_output