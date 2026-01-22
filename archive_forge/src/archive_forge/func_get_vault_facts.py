from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.backup import get_backup_resource_tags
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def get_vault_facts(module, client, vault_name):
    """
    Describes existing vault in an account

    module : AnsibleAWSModule object
    client : boto3 client connection object
    vault_name : Name of the backup vault
    """
    resp = None
    try:
        resp = client.describe_backup_vault(BackupVaultName=vault_name)
    except is_boto3_error_code('AccessDeniedException'):
        module.warn('Access Denied trying to describe backup vault')
    except (BotoCoreError, ClientError) as err:
        module.fail_json_aws(err, msg='Unable to get vault facts')
    if resp:
        if resp.get('BackupVaultArn'):
            resource = resp.get('BackupVaultArn')
            resp['tags'] = get_backup_resource_tags(module, client, resource)
        optional_vals = set(['S3KeyPrefix', 'SnsTopicName', 'SnsTopicARN', 'CloudWatchLogsLogGroupArn', 'CloudWatchLogsRoleArn', 'KmsKeyId'])
        for v in optional_vals - set(resp.keys()):
            resp[v] = None
        return resp
    else:
        return None