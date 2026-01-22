import re
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.waf import MATCH_LOOKUP
from ansible_collections.amazon.aws.plugins.module_utils.waf import get_web_acl_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_regional_rules_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_regional_web_acls_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_rules_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_web_acls_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import run_func_with_change_token_backoff
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def find_and_update_rule(client, module, rule_id):
    rule = get_rule(client, module, rule_id)
    rule_id = rule['RuleId']
    existing_conditions = dict(((condition_type, dict()) for condition_type in MATCH_LOOKUP))
    desired_conditions = dict(((condition_type, dict()) for condition_type in MATCH_LOOKUP))
    all_conditions = dict()
    for condition_type in MATCH_LOOKUP:
        method = 'list_' + MATCH_LOOKUP[condition_type]['method'] + 's'
        all_conditions[condition_type] = dict()
        try:
            paginator = client.get_paginator(method)
            func = paginator.paginate().build_full_result
        except (KeyError, botocore.exceptions.OperationNotPageableError):
            func = getattr(client, method)
        try:
            pred_results = func()[MATCH_LOOKUP[condition_type]['conditionset'] + 's']
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg=f'Could not list {condition_type} conditions')
        for pred in pred_results:
            pred['DataId'] = pred[MATCH_LOOKUP[condition_type]['conditionset'] + 'Id']
            all_conditions[condition_type][pred['Name']] = camel_dict_to_snake_dict(pred)
            all_conditions[condition_type][pred['DataId']] = camel_dict_to_snake_dict(pred)
    for condition in module.params['conditions']:
        desired_conditions[condition['type']][condition['name']] = condition
    reverse_condition_types = dict(((v['type'], k) for k, v in MATCH_LOOKUP.items()))
    for condition in rule['Predicates']:
        existing_conditions[reverse_condition_types[condition['Type']]][condition['DataId']] = camel_dict_to_snake_dict(condition)
    insertions = list()
    deletions = list()
    for condition_type in desired_conditions:
        for condition_name, condition in desired_conditions[condition_type].items():
            if condition_name not in all_conditions[condition_type]:
                module.fail_json(msg=f'Condition {condition_name} of type {condition_type} does not exist')
            condition['data_id'] = all_conditions[condition_type][condition_name]['data_id']
            if condition['data_id'] not in existing_conditions[condition_type]:
                insertions.append(format_for_insertion(condition))
    if module.params['purge_conditions']:
        for condition_type in existing_conditions:
            deletions.extend([format_for_deletion(condition) for condition in existing_conditions[condition_type].values() if not all_conditions[condition_type][condition['data_id']]['name'] in desired_conditions[condition_type]])
    changed = bool(insertions or deletions)
    update = {'RuleId': rule_id, 'Updates': insertions + deletions}
    if changed:
        try:
            run_func_with_change_token_backoff(client, module, update, client.update_rule, wait=True)
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg='Could not update rule conditions')
    return (changed, get_rule(client, module, rule_id))