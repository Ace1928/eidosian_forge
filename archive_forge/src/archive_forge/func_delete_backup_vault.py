from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.backup import get_backup_resource_tags
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def delete_backup_vault(module, client, vault_name):
    """
    Delete a Backup Vault

    module : AnsibleAWSModule object
    client : boto3 client connection object
    vault_name : Backup Vault Name
    """
    try:
        client.delete_backup_vault(BackupVaultName=vault_name)
    except (BotoCoreError, ClientError) as err:
        module.fail_json_aws(err, msg='Failed to delete the Backup Vault')