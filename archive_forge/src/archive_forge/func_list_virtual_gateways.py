from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def list_virtual_gateways(client, module):
    params = dict()
    params['Filters'] = ansible_dict_to_boto3_filter_list(module.params.get('filters'))
    if module.params.get('vpn_gateway_ids'):
        params['VpnGatewayIds'] = module.params.get('vpn_gateway_ids')
    try:
        all_virtual_gateways = client.describe_vpn_gateways(**params)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to list gateways')
    return [camel_dict_to_snake_dict(get_virtual_gateway_info(vgw), ignore_list=['ResourceTags']) for vgw in all_virtual_gateways['VpnGateways']]