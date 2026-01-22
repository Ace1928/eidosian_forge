import secrets
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
def ensure_user_absent(conn, module):
    user = get_matching_user(conn, module, module.params['broker_id'], module.params['username'])
    result = {'changed': False}
    if user is None:
        return result
    if 'Pending' in user and 'PendingChange' in user['Pending'] and (user['Pending']['PendingChange'] == 'DELETE'):
        return result
    result = {'changed': True}
    if module.check_mode:
        return result
    try:
        conn.delete_user(BrokerId=user['BrokerId'], Username=user['Username'])
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg="Couldn't delete user")
    return result