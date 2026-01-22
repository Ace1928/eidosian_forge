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
def delete_unused_regex_pattern(self, regex_pattern_set_id):
    try:
        regex_pattern_set = self.client.get_regex_pattern_set(RegexPatternSetId=regex_pattern_set_id)['RegexPatternSet']
        updates = list()
        for regex_pattern_string in regex_pattern_set['RegexPatternStrings']:
            updates.append({'Action': 'DELETE', 'RegexPatternString': regex_pattern_string})
        run_func_with_change_token_backoff(self.client, self.module, {'RegexPatternSetId': regex_pattern_set_id, 'Updates': updates}, self.client.update_regex_pattern_set)
        run_func_with_change_token_backoff(self.client, self.module, {'RegexPatternSetId': regex_pattern_set_id}, self.client.delete_regex_pattern_set, wait=True)
    except is_boto3_error_code('WAFNonexistentItemException'):
        return
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        self.module.fail_json_aws(e, msg='Could not delete regex pattern')