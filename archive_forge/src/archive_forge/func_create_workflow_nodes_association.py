from __future__ import absolute_import, division, print_function
from ..module_utils.controller_api import ControllerAPIModule
import json
def create_workflow_nodes_association(module, response, workflow_nodes, workflow_id):
    for workflow_node in workflow_nodes:
        workflow_node_fields = {}
        search_fields = {}
        association_fields = {}
        search_fields['workflow_job_template'] = workflow_node_fields['workflow_job_template'] = workflow_id
        if workflow_node['identifier']:
            workflow_node_fields['identifier'] = workflow_node['identifier']
            search_fields['identifier'] = workflow_node['identifier']
        existing_item = module.get_one('workflow_job_template_nodes', **{'data': search_fields})
        if 'state' in workflow_node:
            if workflow_node['state'] == 'absent':
                continue
        if 'related' in workflow_node:
            association_fields = {}
            for association in ('always_nodes', 'success_nodes', 'failure_nodes', 'credentials', 'labels', 'instance_groups'):
                prompt_lookup = ['credentials', 'labels', 'instance_groups']
                if association in workflow_node['related']:
                    id_list = []
                    lookup_data = {}
                    for sub_name in workflow_node['related'][association]:
                        if association in prompt_lookup:
                            endpoint = association
                            if 'organization' in sub_name:
                                lookup_data['organization'] = module.resolve_name_to_id('organizations', sub_name['organization']['name'])
                            lookup_data['name'] = sub_name['name']
                        else:
                            endpoint = 'workflow_job_template_nodes'
                            lookup_data = {'identifier': sub_name['identifier']}
                            lookup_data['workflow_job_template'] = workflow_id
                        sub_obj = module.get_one(endpoint, **{'data': lookup_data})
                        if sub_obj is None:
                            module.fail_json(msg='Could not find {0} entry with name {1}'.format(association, sub_name))
                        id_list.append(sub_obj['id'])
                    if id_list:
                        association_fields[association] = id_list
                    module.create_or_update_if_needed(existing_item, workflow_node_fields, endpoint='workflow_job_template_nodes', item_type='workflow_job_template_node', auto_exit=False, associations=association_fields)