from typing import Union
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
def get_plan_details(module, client, backup_plan_name: str):
    backup_plan_id = _list_backup_plans(client, backup_plan_name)
    if not backup_plan_id:
        return []
    try:
        result = client.get_backup_plan(BackupPlanId=backup_plan_id)
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        module.fail_json_aws(e, msg=f'Failed to describe plan {backup_plan_id}')
    snaked_backup_plan = []
    try:
        resource = result.get('BackupPlanArn', None)
        tag_dict = get_backup_resource_tags(module, client, resource)
        result.update({'tags': tag_dict})
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        module.fail_json_aws(e, msg='Failed to get the backup plan tags')
    snaked_backup_plan.append(camel_dict_to_snake_dict(result, ignore_list='tags'))
    for v in snaked_backup_plan:
        if 'response_metadata' in v:
            del v['response_metadata']
        v['backup_plan_name'] = v['backup_plan']['backup_plan_name']
    return snaked_backup_plan