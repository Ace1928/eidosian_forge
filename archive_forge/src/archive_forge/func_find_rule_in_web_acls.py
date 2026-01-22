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
def find_rule_in_web_acls(client, module, rule_id):
    web_acls_in_use = []
    try:
        if client.__class__.__name__ == 'WAF':
            all_web_acls = list_web_acls_with_backoff(client)
        elif client.__class__.__name__ == 'WAFRegional':
            all_web_acls = list_regional_web_acls_with_backoff(client)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Could not list Web ACLs')
    for web_acl in all_web_acls:
        try:
            web_acl_details = get_web_acl_with_backoff(client, web_acl['WebACLId'])
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg='Could not get Web ACL details')
        if rule_id in [rule['RuleId'] for rule in web_acl_details['Rules']]:
            web_acls_in_use.append(web_acl_details['Name'])
    return web_acls_in_use