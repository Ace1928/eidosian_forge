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
def format_for_update(self, condition_set_id):
    kwargs = dict()
    kwargs['Updates'] = list()
    for filtr in self.module.params.get('filters'):
        if self.type == 'ip':
            if ':' in filtr.get('ip_address'):
                ip_type = 'IPV6'
            else:
                ip_type = 'IPV4'
            condition_insert = {'Type': ip_type, 'Value': filtr.get('ip_address')}
        if self.type == 'geo':
            condition_insert = dict(Type='Country', Value=filtr.get('country'))
        if self.type not in ('ip', 'geo'):
            condition_insert = dict(FieldToMatch=dict(Type=filtr.get('field_to_match').upper()), TextTransformation=filtr.get('transformation', 'none').upper())
            if filtr.get('field_to_match').upper() == 'HEADER':
                if filtr.get('header'):
                    condition_insert['FieldToMatch']['Data'] = filtr.get('header').lower()
                else:
                    self.module.fail_json(msg=str('DATA required when HEADER requested'))
        if self.type == 'byte':
            condition_insert['TargetString'] = filtr.get('target_string')
            condition_insert['PositionalConstraint'] = filtr.get('position')
        if self.type == 'size':
            condition_insert['ComparisonOperator'] = filtr.get('comparison')
            condition_insert['Size'] = filtr.get('size')
        if self.type == 'regex':
            condition_insert['RegexPatternSetId'] = self.ensure_regex_pattern_present(filtr.get('regex_pattern'))['RegexPatternSetId']
        kwargs['Updates'].append({'Action': 'INSERT', self.conditiontuple: condition_insert})
    kwargs[self.conditionsetid] = condition_set_id
    return kwargs