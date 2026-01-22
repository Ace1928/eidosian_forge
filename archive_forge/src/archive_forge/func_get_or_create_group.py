from ansible_collections.amazon.aws.plugins.module_utils.iam import AnsibleIAMError
from ansible_collections.amazon.aws.plugins.module_utils.iam import IAMErrorHandler
from ansible_collections.amazon.aws.plugins.module_utils.iam import convert_managed_policy_names_to_arns
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_iam_group
from ansible_collections.amazon.aws.plugins.module_utils.iam import normalize_iam_group
from ansible_collections.amazon.aws.plugins.module_utils.iam import validate_iam_identifiers
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
@IAMErrorHandler.common_error_handler('create group')
def get_or_create_group(connection, module, group_name, path):
    group = get_iam_group(connection, group_name)
    if group:
        return (False, group)
    params = {'GroupName': group_name}
    if path is not None:
        params['Path'] = path
    if module.check_mode:
        module.exit_json(changed=True, create_params=params)
    group = connection.create_group(aws_retry=True, **params)
    if 'Users' not in group:
        group['Users'] = []
    return (True, group)