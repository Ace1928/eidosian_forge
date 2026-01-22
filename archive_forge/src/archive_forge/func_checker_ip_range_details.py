from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
def checker_ip_range_details():
    results = client.get_checker_ip_ranges()
    results['checker_ip_ranges'] = results['CheckerIpRanges']
    module.deprecate("The 'CamelCase' return values with key 'CheckerIpRanges' is deprecated and will be replaced by 'snake_case' return values with key 'checker_ip_ranges'.  Both case values are returned for now.", date='2025-01-01', collection_name='amazon.aws')
    return results