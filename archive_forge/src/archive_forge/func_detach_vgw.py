import time
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def detach_vgw(client, module, vpn_gateway_id, vpc_id=None):
    params = dict()
    params['VpcId'] = module.params.get('vpc_id')
    try:
        if vpc_id:
            response = client.detach_vpn_gateway(VpnGatewayId=vpn_gateway_id, VpcId=vpc_id, aws_retry=True)
        else:
            response = client.detach_vpn_gateway(VpnGatewayId=vpn_gateway_id, VpcId=params['VpcId'], aws_retry=True)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, 'Failed to detach gateway')
    status_achieved, vgw = wait_for_status(client, module, [vpn_gateway_id], 'detached')
    if not status_achieved:
        module.fail_json(msg='Error waiting for  vpc to detach from vgw - please check the AWS console')
    result = response
    return result