from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def describe_task(self, task_name):
    try:
        response = self.ecs.describe_task_definition(aws_retry=True, taskDefinition=task_name)
        return response['taskDefinition']
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        return None