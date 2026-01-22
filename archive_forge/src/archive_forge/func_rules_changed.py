from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def rules_changed(aws_rules, param_rules, Egress, nacl_id, client, module):
    changed = False
    rules = list()
    for entry in param_rules:
        rules.append(process_rule_entry(entry, Egress))
    if rules == aws_rules:
        return changed
    else:
        removed_rules = [x for x in aws_rules if x not in rules]
        if removed_rules:
            params = dict()
            for rule in removed_rules:
                params['NetworkAclId'] = nacl_id
                params['RuleNumber'] = rule['RuleNumber']
                params['Egress'] = Egress
                delete_network_acl_entry(params, client, module)
            changed = True
        added_rules = [x for x in rules if x not in aws_rules]
        if added_rules:
            for rule in added_rules:
                rule['NetworkAclId'] = nacl_id
                create_network_acl_entry(rule, client, module)
            changed = True
    return changed