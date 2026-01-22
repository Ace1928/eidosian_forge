from __future__ import absolute_import, division, print_function
from ansible.module_utils.common.dict_transformations import _camel_to_snake
from ansible.module_utils._text import to_native
from datetime import datetime, timedelta
def instance_to_dict(self, instance):
    result = dict()
    instance_type = getattr(self.servicebus_models, 'SB{0}'.format(str.capitalize(self.type)))
    attribute_map = instance_type._attribute_map
    for attribute in attribute_map.keys():
        value = getattr(instance, attribute)
        if attribute_map[attribute]['type'] == 'duration':
            if is_valid_timedelta(value):
                key = duration_spec_map.get(attribute) or attribute
                result[key] = int(value.total_seconds())
        elif attribute == 'status':
            result['status'] = _camel_to_snake(value)
        elif isinstance(value, self.servicebus_models.MessageCountDetails):
            result[attribute] = value.as_dict()
        elif isinstance(value, self.servicebus_models.SBSku):
            result[attribute] = value.name.lower()
        elif isinstance(value, datetime):
            result[attribute] = str(value)
        elif isinstance(value, str):
            result[attribute] = to_native(value)
        elif attribute == 'max_size_in_megabytes':
            result['max_size_in_mb'] = value
        else:
            result[attribute] = value
    if self.show_sas_policies and self.type != 'subscription':
        policies = self.get_auth_rules()
        for name in policies.keys():
            policies[name]['keys'] = self.get_sas_key(name)
        result['sas_policies'] = policies
    if self.namespace:
        result['namespace'] = self.namespace
    if self.topic:
        result['topic'] = self.topic
    return result