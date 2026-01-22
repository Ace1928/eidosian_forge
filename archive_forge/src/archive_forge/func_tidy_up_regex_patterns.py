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
def tidy_up_regex_patterns(self, regex_match_set):
    all_regex_match_sets = self.list_conditions()
    all_match_set_patterns = list()
    for rms in all_regex_match_sets:
        all_match_set_patterns.extend((conditiontuple['RegexPatternSetId'] for conditiontuple in self.get_condition_by_id(rms[self.conditionsetid])[self.conditiontuples]))
    for filtr in regex_match_set[self.conditiontuples]:
        if filtr['RegexPatternSetId'] not in all_match_set_patterns:
            self.delete_unused_regex_pattern(filtr['RegexPatternSetId'])