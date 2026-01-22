from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
def reusable_delegation_set_details():
    params = dict()
    if not module.params.get('delegation_set_id'):
        if module.params.get('max_items'):
            params['MaxItems'] = str(module.params.get('max_items'))
        if module.params.get('next_marker'):
            params['Marker'] = module.params.get('next_marker')
        results = client.list_reusable_delegation_sets(**params)
    else:
        params['DelegationSetId'] = module.params.get('delegation_set_id')
        results = client.get_reusable_delegation_set(**params)
    results['delegation_sets'] = results['DelegationSets']
    module.deprecate("The 'CamelCase' return values with key 'DelegationSets' is deprecated and will be replaced by 'snake_case' return values with key 'delegation_sets'.  Both case values are returned for now.", date='2025-01-01', collection_name='amazon.aws')
    return results