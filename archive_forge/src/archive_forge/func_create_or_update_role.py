import json
from ansible_collections.amazon.aws.plugins.module_utils.arn import validate_aws_arn
from ansible_collections.amazon.aws.plugins.module_utils.iam import AnsibleIAMError
from ansible_collections.amazon.aws.plugins.module_utils.iam import IAMErrorHandler
from ansible_collections.amazon.aws.plugins.module_utils.iam import add_role_to_iam_instance_profile
from ansible_collections.amazon.aws.plugins.module_utils.iam import convert_managed_policy_names_to_arns
from ansible_collections.amazon.aws.plugins.module_utils.iam import create_iam_instance_profile
from ansible_collections.amazon.aws.plugins.module_utils.iam import delete_iam_instance_profile
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_iam_role
from ansible_collections.amazon.aws.plugins.module_utils.iam import list_iam_instance_profiles
from ansible_collections.amazon.aws.plugins.module_utils.iam import list_iam_role_attached_policies
from ansible_collections.amazon.aws.plugins.module_utils.iam import normalize_iam_role
from ansible_collections.amazon.aws.plugins.module_utils.iam import remove_role_from_iam_instance_profile
from ansible_collections.amazon.aws.plugins.module_utils.iam import validate_iam_identifiers
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def create_or_update_role(module, client):
    check_mode = module.check_mode
    wait = module.params.get('wait')
    wait_timeout = module.params.get('wait_timeout')
    role_name = module.params.get('name')
    create_instance_profile = module.params.get('create_instance_profile')
    path = module.params.get('path')
    purge_policies = module.params.get('purge_policies')
    managed_policies = module.params.get('managed_policies')
    if managed_policies:
        managed_policies = convert_managed_policy_names_to_arns(client, managed_policies)
    changed = False
    role = get_iam_role(client, role_name)
    if role is None:
        role = create_basic_role(module, client)
        wait_iam_exists(client, check_mode, role_name, wait, wait_timeout)
        changed = True
    else:
        changed = update_basic_role(module, client, role_name, role)
        wait_iam_exists(client, check_mode, role_name, wait, wait_timeout)
    if create_instance_profile:
        changed |= create_instance_profiles(client, check_mode, role_name, path)
        wait_iam_exists(client, check_mode, role_name, wait, wait_timeout)
    changed |= update_managed_policies(client, module.check_mode, role_name, managed_policies, purge_policies)
    wait_iam_exists(client, check_mode, role_name, wait, wait_timeout)
    role = get_iam_role(client, role_name)
    role['AttachedPolicies'] = list_iam_role_attached_policies(client, role_name)
    camel_role = normalize_iam_role(role, _v7_compat=True)
    module.exit_json(changed=changed, iam_role=camel_role, **camel_role)