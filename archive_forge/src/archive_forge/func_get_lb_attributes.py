from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_lb_attributes(connection, load_balancer_name):
    attributes = connection.describe_load_balancer_attributes(aws_retry=True, LoadBalancerName=load_balancer_name).get('LoadBalancerAttributes', {})
    return camel_dict_to_snake_dict(attributes)