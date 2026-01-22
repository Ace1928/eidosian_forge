import time
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_vgw(client, module):
    params = dict()
    params['Type'] = module.params.get('type')
    tags = module.params.get('tags') or {}
    tags['Name'] = module.params.get('name')
    params['TagSpecifications'] = boto3_tag_specifications(tags, ['vpn-gateway'])
    if module.params.get('asn'):
        params['AmazonSideAsn'] = module.params.get('asn')
    try:
        response = client.create_vpn_gateway(aws_retry=True, **params)
        get_waiter(client, 'vpn_gateway_exists').wait(VpnGatewayIds=[response['VpnGateway']['VpnGatewayId']])
    except botocore.exceptions.WaiterError as e:
        module.fail_json_aws(e, msg=f'Failed to wait for Vpn Gateway {response['VpnGateway']['VpnGatewayId']} to be available')
    except is_boto3_error_code('VpnGatewayLimitExceeded') as e:
        module.fail_json_aws(e, msg='Too many VPN gateways exist in this account.')
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to create gateway')
    result = response
    return result