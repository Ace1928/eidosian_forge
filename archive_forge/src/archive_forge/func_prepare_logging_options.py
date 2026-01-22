import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def prepare_logging_options(module):
    l_params = {}
    logging = module.params['logging'] or {}
    if logging.get('cloudwatch'):
        l_params['CloudWatchLogs'] = {'Enabled': module.params['logging']['cloudwatch'].get('enabled'), 'LogGroup': module.params['logging']['cloudwatch'].get('log_group')}
    else:
        l_params['CloudWatchLogs'] = {'Enabled': False}
    if logging.get('firehose'):
        l_params['Firehose'] = {'Enabled': module.params['logging']['firehose'].get('enabled'), 'DeliveryStream': module.params['logging']['firehose'].get('delivery_stream')}
    else:
        l_params['Firehose'] = {'Enabled': False}
    if logging.get('s3'):
        l_params['S3'] = {'Enabled': module.params['logging']['s3'].get('enabled'), 'Bucket': module.params['logging']['s3'].get('bucket'), 'Prefix': module.params['logging']['s3'].get('prefix')}
    else:
        l_params['S3'] = {'Enabled': False}
    return {'LoggingInfo': {'BrokerLogs': l_params}}