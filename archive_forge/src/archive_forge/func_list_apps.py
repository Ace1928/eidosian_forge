from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def list_apps(ebs, app_name, module):
    try:
        if app_name is not None:
            apps = ebs.describe_applications(ApplicationNames=[app_name])
        else:
            apps = ebs.describe_applications()
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Could not describe application')
    return apps.get('Applications', [])