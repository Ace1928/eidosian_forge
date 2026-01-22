from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_aws_account_info
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
@AWSRetry.exponential_backoff()
def describe_replication_group_with_backoff(client, replication_group_id):
    try:
        response = client.describe_replication_groups(ReplicationGroupId=replication_group_id)
    except is_boto3_error_code('ReplicationGroupNotFoundFault'):
        return None
    return response['ReplicationGroups'][0]