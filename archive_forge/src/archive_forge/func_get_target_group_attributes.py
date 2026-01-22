from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_target_group_attributes(target_group_arn):
    try:
        target_group_attributes = boto3_tag_list_to_ansible_dict(client.describe_target_group_attributes(TargetGroupArn=target_group_arn)['Attributes'])
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to describe target group attributes')
    return dict(((k.replace('.', '_'), v) for k, v in target_group_attributes.items()))