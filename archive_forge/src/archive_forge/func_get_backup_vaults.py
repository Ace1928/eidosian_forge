from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.backup import get_backup_resource_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
def get_backup_vaults(connection, module):
    all_backup_vaults = []
    try:
        result = connection.get_paginator('list_backup_vaults')
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to get the backup vaults.')
    for backup_vault in result.paginate():
        all_backup_vaults.extend(list_backup_vaults(backup_vault))
    return all_backup_vaults