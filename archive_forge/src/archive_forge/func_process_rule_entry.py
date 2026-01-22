from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def process_rule_entry(entry, Egress):
    params = dict()
    params['RuleNumber'] = entry[0]
    params['Protocol'] = str(PROTOCOL_NUMBERS[entry[1]])
    params['RuleAction'] = entry[2]
    params['Egress'] = Egress
    if is_ipv6(entry[3]):
        params['Ipv6CidrBlock'] = entry[3]
    else:
        params['CidrBlock'] = entry[3]
    if icmp_present(entry):
        params['IcmpTypeCode'] = {'Type': int(entry[4]), 'Code': int(entry[5])}
    elif entry[6] or entry[7]:
        params['PortRange'] = {'From': entry[6], 'To': entry[7]}
    return params