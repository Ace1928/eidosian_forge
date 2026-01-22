import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def prepare_open_monitoring_options(module):
    m_params = {}
    open_monitoring = module.params['open_monitoring'] or {}
    m_params['OpenMonitoring'] = {'Prometheus': {'JmxExporter': {'EnabledInBroker': open_monitoring.get('jmx_exporter', False)}, 'NodeExporter': {'EnabledInBroker': open_monitoring.get('node_exporter', False)}}}
    return m_params