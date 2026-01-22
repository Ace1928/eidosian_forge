from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.iam import AnsibleIAMError
from ansible_collections.amazon.aws.plugins.module_utils.iam import IAMErrorHandler
from ansible_collections.amazon.aws.plugins.module_utils.iam import convert_managed_policy_names_to_arns
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_iam_user
from ansible_collections.amazon.aws.plugins.module_utils.iam import normalize_iam_user
from ansible_collections.amazon.aws.plugins.module_utils.iam import validate_iam_identifiers
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def create_or_update_user(connection, module):
    user_name = module.params.get('name')
    changed = False
    new_user = False
    user = get_iam_user(connection, user_name)
    boundary = module.params.get('boundary')
    if boundary:
        boundary = convert_managed_policy_names_to_arns(connection, [module.params.get('boundary')])[0]
    if user is None:
        user = create_user(connection, module, user_name, module.params.get('path'), boundary, module.params.get('tags'))
        changed = True
        wait_iam_exists(connection, module)
        new_user = True
    profile_changed, login_profile = ensure_login_profile(connection, module.check_mode, user_name, module.params.get('password'), module.params.get('update_password'), module.params.get('password_reset_required'), new_user)
    changed |= profile_changed
    changed |= remove_login_profile(connection, module.check_mode, user_name, module.params.get('remove_password'), new_user)
    changed |= ensure_permissions_boundary(connection, module.check_mode, user, user_name, boundary)
    changed |= ensure_path(connection, module.check_mode, user, user_name, module.params.get('path'))
    changed |= ensure_managed_policies(connection, module.check_mode, user_name, module.params.get('managed_policies'), module.params.get('purge_policies'))
    changed |= ensure_user_tags(connection, module.check_mode, user, user_name, module.params.get('tags'), module.params.get('purge_tags'))
    if module.check_mode:
        module.exit_json(changed=changed)
    user = get_iam_user(connection, user_name)
    if changed and login_profile:
        user['password_reset_required'] = login_profile.get('LoginProfile', {}).get('PasswordResetRequired', False)
    try:
        policies = {'attached_policies': _list_attached_policies(connection, user_name)}
        user.update(camel_dict_to_snake_dict(policies))
    except AnsibleIAMError as e:
        module.warn(f'Failed to list attached policies - {str(e.exception)}')
        pass
    module.exit_json(changed=changed, iam_user={'user': user}, user=user)