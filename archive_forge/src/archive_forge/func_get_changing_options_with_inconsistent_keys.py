from time import sleep
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.botocore import get_boto3_client_method_parameters
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.rds import arg_spec_to_rds_params
from ansible_collections.amazon.aws.plugins.module_utils.rds import call_method
from ansible_collections.amazon.aws.plugins.module_utils.rds import compare_iam_roles
from ansible_collections.amazon.aws.plugins.module_utils.rds import ensure_tags
from ansible_collections.amazon.aws.plugins.module_utils.rds import get_final_identifier
from ansible_collections.amazon.aws.plugins.module_utils.rds import get_rds_method_attribute
from ansible_collections.amazon.aws.plugins.module_utils.rds import get_tags
from ansible_collections.amazon.aws.plugins.module_utils.rds import update_iam_roles
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
def get_changing_options_with_inconsistent_keys(modify_params, instance, purge_cloudwatch_logs, purge_security_groups):
    changing_params = {}
    current_options = get_current_attributes_with_inconsistent_keys(instance)
    for option in current_options:
        current_option = current_options[option]
        desired_option = modify_params.pop(option, None)
        if desired_option is None:
            continue
        if isinstance(current_option, list):
            if isinstance(desired_option, list):
                if set(desired_option) < set(current_option) and option in ['DBSecurityGroups', 'VpcSecurityGroupIds'] and purge_security_groups:
                    changing_params[option] = desired_option
                elif set(desired_option) <= set(current_option):
                    continue
            elif isinstance(desired_option, string_types):
                if desired_option in current_option:
                    continue
        if option != 'ProcessorFeatures' and current_option == desired_option:
            continue
        if option == 'ProcessorFeatures' and current_option == boto3_tag_list_to_ansible_dict(desired_option, 'Name', 'Value'):
            continue
        if option == 'ProcessorFeatures' and desired_option == []:
            changing_params['UseDefaultProcessorFeatures'] = True
        elif option == 'CloudwatchLogsExportConfiguration':
            current_option = set(current_option.get('LogTypesToEnable', []))
            desired_option = set(desired_option)
            format_option = {'EnableLogTypes': [], 'DisableLogTypes': []}
            format_option['EnableLogTypes'] = list(desired_option.difference(current_option))
            if purge_cloudwatch_logs:
                format_option['DisableLogTypes'] = list(current_option.difference(desired_option))
            if format_option['EnableLogTypes'] or format_option['DisableLogTypes']:
                changing_params[option] = format_option
        elif option in ['DBSecurityGroups', 'VpcSecurityGroupIds']:
            if purge_security_groups:
                changing_params[option] = desired_option
            else:
                changing_params[option] = list(set(current_option) | set(desired_option))
        else:
            changing_params[option] = desired_option
    return changing_params