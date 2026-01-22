from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_lb(connection, load_balancer_name):
    try:
        return connection.describe_load_balancers(aws_retry=True, LoadBalancerNames=[load_balancer_name])['LoadBalancerDescriptions'][0]
    except is_boto3_error_code('LoadBalancerNotFound'):
        return []