from typing import Union
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
def get_backup_resource_tags(module, backup_client, resource):
    try:
        response = backup_client.list_tags(ResourceArn=resource)
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        module.fail_json_aws(e, msg=f'Failed to list tags on the resource {resource}')
    return response['Tags']