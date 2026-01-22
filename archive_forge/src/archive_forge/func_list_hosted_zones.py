from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
def list_hosted_zones():
    params = dict()
    if module.params.get('max_items'):
        params['PaginationConfig'] = dict(MaxItems=module.params.get('max_items'))
    if module.params.get('next_marker'):
        params['Marker'] = module.params.get('next_marker')
    if module.params.get('delegation_set_id'):
        params['DelegationSetId'] = module.params.get('delegation_set_id')
    zones = _paginated_result('list_hosted_zones', **params)['HostedZones']
    snaked_zones = [camel_dict_to_snake_dict(zone) for zone in zones]
    module.deprecate("The 'CamelCase' return values with key 'HostedZones' and 'list' are deprecated and will be replaced by 'snake_case' return values with key 'hosted_zones'.  Both case values are returned for now.", date='2025-01-01', collection_name='amazon.aws')
    return {'HostedZones': zones, 'list': zones, 'hosted_zones': snaked_zones}