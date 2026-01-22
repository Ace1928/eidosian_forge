import re
from ipaddress import ip_network
from time import sleep
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import describe_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def find_igw(connection, module, vpc_id):
    """
    Finds the Internet gateway for the given VPC ID.
    """
    filters = ansible_dict_to_boto3_filter_list({'attachment.vpc-id': vpc_id})
    try:
        igw = describe_igws_with_backoff(connection, Filters=filters)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg=f'No IGW found for VPC {vpc_id}')
    if len(igw) == 1:
        return igw[0]['InternetGatewayId']
    elif len(igw) == 0:
        module.fail_json(msg=f'No IGWs found for VPC {vpc_id}')
    else:
        module.fail_json(msg=f'Multiple IGWs found for VPC {vpc_id}')