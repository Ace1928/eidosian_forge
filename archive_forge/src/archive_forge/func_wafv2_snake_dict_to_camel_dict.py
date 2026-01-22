from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def wafv2_snake_dict_to_camel_dict(a):
    if not isinstance(a, dict):
        return a
    retval = {}
    for item in a.keys():
        if isinstance(a.get(item), dict):
            if 'Ip' in item:
                retval[item.replace('Ip', 'IP')] = wafv2_snake_dict_to_camel_dict(a.get(item))
            elif 'Arn' == item:
                retval['ARN'] = wafv2_snake_dict_to_camel_dict(a.get(item))
            else:
                retval[item] = wafv2_snake_dict_to_camel_dict(a.get(item))
        elif isinstance(a.get(item), list):
            retval[item] = []
            for idx in range(len(a.get(item))):
                retval[item].append(wafv2_snake_dict_to_camel_dict(a.get(item)[idx]))
        elif 'Ip' in item:
            retval[item.replace('Ip', 'IP')] = a.get(item)
        elif 'Arn' == item:
            retval['ARN'] = a.get(item)
        else:
            retval[item] = a.get(item)
    return retval