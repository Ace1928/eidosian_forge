import re
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_regional_rules_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_regional_web_acls_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_rules_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_web_acls_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import run_func_with_change_token_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def ensure_web_acl_absent(client, module):
    web_acl_id = get_web_acl_by_name(client, module, module.params['name'])
    if web_acl_id:
        web_acl = get_web_acl(client, module, web_acl_id)
        if web_acl['Rules']:
            remove_rules_from_web_acl(client, module, web_acl_id)
        try:
            run_func_with_change_token_backoff(client, module, {'WebACLId': web_acl_id}, client.delete_web_acl, wait=True)
            return (True, {})
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg='Could not delete Web ACL')
    return (False, {})