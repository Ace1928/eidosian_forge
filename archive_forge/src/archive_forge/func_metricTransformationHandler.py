from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
def metricTransformationHandler(metricTransformations, originMetricTransformations=None):
    if originMetricTransformations:
        change = False
        originMetricTransformations = camel_dict_to_snake_dict(originMetricTransformations)
        for item in ['default_value', 'metric_name', 'metric_namespace', 'metric_value']:
            if metricTransformations.get(item) != originMetricTransformations.get(item):
                change = True
    else:
        change = True
    defaultValue = metricTransformations.get('default_value')
    if isinstance(defaultValue, int) or isinstance(defaultValue, float):
        retval = [{'metricName': metricTransformations.get('metric_name'), 'metricNamespace': metricTransformations.get('metric_namespace'), 'metricValue': metricTransformations.get('metric_value'), 'defaultValue': defaultValue}]
    else:
        retval = [{'metricName': metricTransformations.get('metric_name'), 'metricNamespace': metricTransformations.get('metric_namespace'), 'metricValue': metricTransformations.get('metric_value')}]
    return (retval, change)