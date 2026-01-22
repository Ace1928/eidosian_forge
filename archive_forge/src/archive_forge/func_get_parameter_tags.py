import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.base import BaseWaiterFactory
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_parameter_tags(client, module, parameter_name):
    try:
        tags = client.list_tags_for_resource(aws_retry=True, ResourceType='Parameter', ResourceId=parameter_name)['TagList']
        tags_dict = boto3_tag_list_to_ansible_dict(tags)
        return tags_dict
    except (BotoCoreError, ClientError) as e:
        module.fail_json_aws(e, msg='Unable to retrieve parameter tags')