from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def describe_app(ebs, app_name, module):
    apps = list_apps(ebs, app_name, module)
    return None if len(apps) != 1 else apps[0]