from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
def set_properties_if_changed(self):
    diff = {'before': {'extra_zfs_properties': {}}, 'after': {'extra_zfs_properties': {}}}
    current_properties = self.get_current_properties()
    for prop, value in self.properties.items():
        current_value = current_properties.get(prop, None)
        if current_value != value:
            self.set_property(prop, value)
            diff['before']['extra_zfs_properties'][prop] = current_value
            diff['after']['extra_zfs_properties'][prop] = value
    if self.module.check_mode:
        return diff
    updated_properties = self.get_current_properties()
    for prop in self.properties:
        value = updated_properties.get(prop, None)
        if value is None:
            self.module.fail_json(msg='zfsprop was not present after being successfully set: %s' % prop)
        if current_properties.get(prop, None) != value:
            self.changed = True
        if prop in diff['after']['extra_zfs_properties']:
            diff['after']['extra_zfs_properties'][prop] = value
    return diff