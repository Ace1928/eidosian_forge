import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def dx_gateway_info(client, gateway_id, module):
    try:
        resp = client.describe_direct_connect_gateways(directConnectGatewayId=gateway_id)
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        module.fail_json_aws(e, msg='Failed to fetch gateway information.')
    if resp['directConnectGateways']:
        return resp['directConnectGateways'][0]