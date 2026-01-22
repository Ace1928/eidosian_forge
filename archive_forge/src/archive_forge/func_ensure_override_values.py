from __future__ import absolute_import, division, print_function
from ansible_collections.theforeman.foreman.plugins.module_utils.foreman_helper import ForemanEntityAnsibleModule, parameter_value_to_str
def ensure_override_values(self, entity, expected_override_values):
    if expected_override_values is not None:
        parameter_type = entity.get('parameter_type', 'string')
        scope = {'smart_class_parameter_id': entity['id']}
        if not self.desired_absent:
            current_override_values = {override_value['match']: override_value for override_value in entity.get('override_values', [])}
            desired_override_values = {override_value['match']: override_value for override_value in expected_override_values}
            for match in desired_override_values:
                desired_override_value = desired_override_values[match]
                if 'value' in desired_override_value:
                    desired_override_value['value'] = parameter_value_to_str(desired_override_value['value'], parameter_type)
                current_override_value = current_override_values.pop(match, None)
                if current_override_value:
                    current_override_value['value'] = parameter_value_to_str(current_override_value['value'], parameter_type)
                self.ensure_entity('override_values', desired_override_value, current_override_value, state='present', foreman_spec=override_value_foreman_spec, params=scope)
            for current_override_value in current_override_values.values():
                self.ensure_entity('override_values', None, current_override_value, state='absent', foreman_spec=override_value_foreman_spec, params=scope)