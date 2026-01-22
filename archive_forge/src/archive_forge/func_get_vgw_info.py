import time
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_vgw_info(vgws):
    if not isinstance(vgws, list):
        return
    for vgw in vgws:
        vgw_info = {'id': vgw['VpnGatewayId'], 'type': vgw['Type'], 'state': vgw['State'], 'vpc_id': None, 'tags': dict()}
        if vgw['Tags']:
            vgw_info['tags'] = boto3_tag_list_to_ansible_dict(vgw['Tags'])
        if len(vgw['VpcAttachments']) != 0 and vgw['VpcAttachments'][0]['State'] == 'attached':
            vgw_info['vpc_id'] = vgw['VpcAttachments'][0]['VpcId']
        return vgw_info