import json
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
def describe_stack_change_sets(self, stack_name):
    changes = []
    try:
        change_sets = self.list_stack_change_sets_with_backoff(stack_name)
        for item in change_sets:
            changes.append(self.describe_stack_change_set_with_backoff(StackName=stack_name, ChangeSetName=item['ChangeSetName']))
        return changes
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        self.module.fail_json_aws(e, msg='Error describing stack change sets for stack ' + stack_name)