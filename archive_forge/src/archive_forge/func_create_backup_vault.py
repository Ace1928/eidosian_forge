from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.backup import get_backup_resource_tags
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def create_backup_vault(module, client, params):
    """
    Creates a Backup Vault

    module : AnsibleAWSModule object
    client : boto3 client connection object
    params : The parameters to create a backup vault
    """
    resp = {}
    params = {k: v for k, v in params.items() if v is not None}
    try:
        resp = client.create_backup_vault(**params)
    except (BotoCoreError, ClientError) as err:
        module.fail_json_aws(err, msg='Failed to create Backup Vault')
    return resp