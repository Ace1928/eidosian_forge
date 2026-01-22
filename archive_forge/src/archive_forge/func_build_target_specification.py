from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def build_target_specification(target_tracking_config):
    targetTrackingConfig = dict()
    if target_tracking_config.get('target_value'):
        targetTrackingConfig['TargetValue'] = target_tracking_config['target_value']
    if target_tracking_config.get('disable_scalein'):
        targetTrackingConfig['DisableScaleIn'] = target_tracking_config['disable_scalein']
    else:
        targetTrackingConfig['DisableScaleIn'] = False
    if target_tracking_config['predefined_metric_spec'] is not None:
        targetTrackingConfig['PredefinedMetricSpecification'] = dict()
        if target_tracking_config['predefined_metric_spec'].get('predefined_metric_type'):
            targetTrackingConfig['PredefinedMetricSpecification']['PredefinedMetricType'] = target_tracking_config['predefined_metric_spec']['predefined_metric_type']
        if target_tracking_config['predefined_metric_spec'].get('resource_label'):
            targetTrackingConfig['PredefinedMetricSpecification']['ResourceLabel'] = target_tracking_config['predefined_metric_spec']['resource_label']
    elif target_tracking_config['customized_metric_spec'] is not None:
        targetTrackingConfig['CustomizedMetricSpecification'] = dict()
        if target_tracking_config['customized_metric_spec'].get('metric_name'):
            targetTrackingConfig['CustomizedMetricSpecification']['MetricName'] = target_tracking_config['customized_metric_spec']['metric_name']
        if target_tracking_config['customized_metric_spec'].get('namespace'):
            targetTrackingConfig['CustomizedMetricSpecification']['Namespace'] = target_tracking_config['customized_metric_spec']['namespace']
        if target_tracking_config['customized_metric_spec'].get('dimensions'):
            targetTrackingConfig['CustomizedMetricSpecification']['Dimensions'] = target_tracking_config['customized_metric_spec']['dimensions']
        if target_tracking_config['customized_metric_spec'].get('statistic'):
            targetTrackingConfig['CustomizedMetricSpecification']['Statistic'] = target_tracking_config['customized_metric_spec']['statistic']
        if target_tracking_config['customized_metric_spec'].get('unit'):
            targetTrackingConfig['CustomizedMetricSpecification']['Unit'] = target_tracking_config['customized_metric_spec']['unit']
    return targetTrackingConfig