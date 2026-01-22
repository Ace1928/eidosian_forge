from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def nested_byte_values_to_strings(rule, keyname):
    """
    currently valid nested byte values in statements array are
        - OrStatement
        - AndStatement
        - NotStatement
    """
    if rule.get('Statement', {}).get(keyname):
        for idx in range(len(rule.get('Statement', {}).get(keyname, {}).get('Statements'))):
            if rule['Statement'][keyname]['Statements'][idx].get('ByteMatchStatement'):
                rule['Statement'][keyname]['Statements'][idx]['ByteMatchStatement']['SearchString'] = rule.get('Statement').get(keyname).get('Statements')[idx].get('ByteMatchStatement').get('SearchString').decode('utf-8')
    return rule