from ansible_collections.amazon.aws.plugins.module_utils.ec2 import get_ec2_security_group_ids_from_names
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def ecs_task_long_format_enabled(self):
    account_support = self.ecs.list_account_settings(name='taskLongArnFormat', effectiveSettings=True)
    return account_support['settings'][0]['value'] == 'enabled'