from typing import Union
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
def get_selection_details(module, client, plan_name: str, selection_name: Union[str, list]):
    result = []
    plan = get_plan_details(module, client, plan_name)
    if not plan:
        module.fail_json(msg=f'The backup plan {plan_name} does not exist. Please create one first.')
    plan_id = plan[0]['backup_plan_id']
    selection_list = _list_backup_selections(client, module, plan_id)
    if selection_name:
        for selection in selection_list:
            if isinstance(selection_name, list):
                for name in selection_name:
                    if selection['SelectionName'] == name:
                        selection_id = selection['SelectionId']
                        selection_info = _get_backup_selection(client, module, plan_id, selection_id)
                        result.append(selection_info)
            if isinstance(selection_name, str):
                if selection['SelectionName'] == selection_name:
                    selection_id = selection['SelectionId']
                    result.append(_get_backup_selection(client, module, plan_id, selection_id))
                    break
    else:
        for selection in selection_list:
            selection_id = selection['SelectionId']
            result.append(_get_backup_selection(client, module, plan_id, selection_id))
    for v in result:
        if 'ResponseMetadata' in v:
            del v['ResponseMetadata']
        if 'BackupSelection' in v:
            for backup_selection_key in v['BackupSelection']:
                v[backup_selection_key] = v['BackupSelection'][backup_selection_key]
        del v['BackupSelection']
    return result