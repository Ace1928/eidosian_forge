import json
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.backup import get_plan_details
from ansible_collections.amazon.aws.plugins.module_utils.backup import get_selection_details
from ansible_collections.amazon.aws.plugins.module_utils.core import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import AWSRetry
def check_for_update(current_selection, backup_selection_data, iam_role_arn):
    update_needed = False
    if current_selection[0].get('IamRoleArn', None) != iam_role_arn:
        update_needed = True
    fields_to_check = ['Resources', 'ListOfTags', 'NotResources', 'Conditions']
    for field_name in fields_to_check:
        field_value_from_aws = json.dumps(current_selection[0].get(field_name, []), sort_keys=True)
        new_field_value = json.dumps(backup_selection_data.get(field_name, []), sort_keys=True)
        if new_field_value != field_value_from_aws:
            if field_name != 'Conditions':
                update_needed = True
            elif not (field_value_from_aws == '{"StringEquals": [], "StringLike": [], "StringNotEquals": [], "StringNotLike": []}' and new_field_value == '[]'):
                update_needed = True
    return update_needed