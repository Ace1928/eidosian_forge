from ansible_collections.amazon.aws.plugins.module_utils.iam import AnsibleIAMError
from ansible_collections.amazon.aws.plugins.module_utils.iam import IAMErrorHandler
from ansible_collections.amazon.aws.plugins.module_utils.iam import convert_managed_policy_names_to_arns
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_iam_group
from ansible_collections.amazon.aws.plugins.module_utils.iam import normalize_iam_group
from ansible_collections.amazon.aws.plugins.module_utils.iam import validate_iam_identifiers
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
def create_or_update_group(connection, module):
    changed, group_info = get_or_create_group(connection, module, module.params['name'], module.params['path'])
    changed |= ensure_path(connection, module, group_info, module.params['path'])
    changed |= ensure_managed_policies(connection, module, group_info, module.params['managed_policies'], module.params['purge_policies'])
    changed |= ensure_group_members(connection, module, group_info, module.params['users'], module.params['purge_users'])
    if module.check_mode:
        module.exit_json(changed=changed)
    group_info = get_iam_group(connection, module.params['name'])
    policies = get_attached_policy_list(connection, module, module.params['name'])
    group_info['AttachedPolicies'] = policies
    module.exit_json(changed=changed, iam_group=normalize_iam_group(group_info))