from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.waf import MATCH_LOOKUP
from ansible_collections.amazon.aws.plugins.module_utils.waf import get_rule_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_regional_rules_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_rules_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import run_func_with_change_token_backoff
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def find_condition_in_rules(self, condition_set_id):
    rules_in_use = []
    try:
        if self.client.__class__.__name__ == 'WAF':
            all_rules = list_rules_with_backoff(self.client)
        elif self.client.__class__.__name__ == 'WAFRegional':
            all_rules = list_regional_rules_with_backoff(self.client)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        self.module.fail_json_aws(e, msg='Could not list rules')
    for rule in all_rules:
        try:
            rule_details = get_rule_with_backoff(self.client, rule['RuleId'])
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            self.module.fail_json_aws(e, msg='Could not get rule details')
        if condition_set_id in [predicate['DataId'] for predicate in rule_details['Predicates']]:
            rules_in_use.append(rule_details['Name'])
    return rules_in_use