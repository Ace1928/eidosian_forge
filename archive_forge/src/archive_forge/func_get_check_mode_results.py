from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_check_mode_results(connection, module_params, vpn_connection_id=None, current_state=None):
    """Returns the changes that would be made to a VPN Connection"""
    state = module_params.get('state')
    if state == 'absent':
        if vpn_connection_id:
            return (True, {})
        else:
            return (False, {})
    changed = False
    results = {'customer_gateway_configuration': '', 'customer_gateway_id': module_params.get('customer_gateway_id'), 'vpn_gateway_id': module_params.get('vpn_gateway_id'), 'transit_gateway_id': module_params.get('transit_gateway_id'), 'options': {'static_routes_only': module_params.get('static_only')}, 'routes': [module_params.get('routes')]}
    present_tags = module_params.get('tags')
    if present_tags is None:
        pass
    elif current_state and 'Tags' in current_state:
        current_tags = boto3_tag_list_to_ansible_dict(current_state['Tags'])
        tags_to_add, tags_to_remove = compare_aws_tags(current_tags, present_tags, module_params.get('purge_tags'))
        changed |= bool(tags_to_remove) or bool(tags_to_add)
        if module_params.get('purge_tags'):
            current_tags = {}
        current_tags.update(present_tags)
        results['tags'] = current_tags
    elif module_params.get('tags'):
        changed = True
    if present_tags:
        results['tags'] = present_tags
    present_routes = module_params.get('routes')
    if current_state and 'Routes' in current_state:
        current_routes = [route['DestinationCidrBlock'] for route in current_state['Routes']]
        if module_params.get('purge_routes'):
            if set(current_routes) != set(present_routes):
                changed = True
        elif set(present_routes) != set(current_routes):
            if not set(present_routes) < set(current_routes):
                changed = True
            present_routes.extend([route for route in current_routes if route not in present_routes])
    elif module_params.get('routes'):
        changed = True
    results['routes'] = [{'destination_cidr_block': cidr, 'state': 'available'} for cidr in present_routes]
    if vpn_connection_id:
        results['vpn_connection_id'] = vpn_connection_id
    else:
        changed = True
        results['vpn_connection_id'] = 'vpn-XXXXXXXX'
    return (changed, results)