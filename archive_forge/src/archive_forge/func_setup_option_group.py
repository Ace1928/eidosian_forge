from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.rds import get_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def setup_option_group(client, module):
    results = []
    changed = False
    to_be_added = None
    to_be_removed = None
    existing_option_group = get_option_group(client, module)
    if existing_option_group:
        results = existing_option_group
        changed |= update_tags(client, module, existing_option_group)
        if module.params.get('options'):
            update_required = match_option_group_options(client, module)
            if update_required:
                to_be_added, to_be_removed = compare_option_group(client, module)
            if to_be_added or update_required:
                changed |= create_option_group_options(client, module)
            if to_be_removed:
                changed |= remove_option_group_options(client, module, to_be_removed)
            if changed:
                results = get_option_group(client, module)
        else:
            current_option_group = get_option_group(client, module)
            if current_option_group['options'] != []:
                options_to_remove = []
                for option in current_option_group['options']:
                    options_to_remove.append(option['option_name'])
                changed |= remove_option_group_options(client, module, options_to_remove)
            if changed:
                results = get_option_group(client, module)
    else:
        changed = create_option_group(client, module)
        if module.params.get('options'):
            changed = create_option_group_options(client, module)
        results = get_option_group(client, module)
    return (changed, results)