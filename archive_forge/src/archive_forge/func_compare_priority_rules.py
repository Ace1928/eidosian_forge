from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def compare_priority_rules(existing_rules, requested_rules, purge_rules, state):
    diff = False
    existing_rules = sorted(existing_rules, key=lambda k: k['Priority'])
    existing_rules = byte_values_to_strings_before_compare(existing_rules)
    requested_rules = sorted(requested_rules, key=lambda k: k['Priority'])
    if purge_rules and state == 'present':
        merged_rules = requested_rules
        if len(existing_rules) == len(requested_rules):
            for idx in range(len(existing_rules)):
                if existing_rules[idx] != requested_rules[idx]:
                    diff = True
                    break
        else:
            diff = True
    else:
        merged_rules = []
        ex_idx_pop = []
        for existing_idx in range(len(existing_rules)):
            for requested_idx in range(len(requested_rules)):
                if existing_rules[existing_idx].get('Priority') == requested_rules[requested_idx].get('Priority'):
                    if state == 'present':
                        ex_idx_pop.append(existing_idx)
                        if existing_rules[existing_idx] != requested_rules[requested_idx]:
                            diff = True
                    elif existing_rules[existing_idx] == requested_rules[requested_idx]:
                        ex_idx_pop.append(existing_idx)
                        diff = True
        prev_count = len(existing_rules)
        for idx in ex_idx_pop:
            existing_rules.pop(idx)
        if state == 'present':
            merged_rules = existing_rules + requested_rules
            if len(merged_rules) != prev_count:
                diff = True
        else:
            merged_rules = existing_rules
    return (diff, merged_rules)