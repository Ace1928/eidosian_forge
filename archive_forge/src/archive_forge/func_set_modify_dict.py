from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def set_modify_dict(self, current, after_create=False):
    """Fill modify dict with changes"""
    octal_value = current.get('unix_permissions') if current else None
    if self.parameters.get('unix_permissions') is not None and self.na_helper.compare_chmod_value(octal_value, self.parameters['unix_permissions']):
        del self.parameters['unix_permissions']
    auto_delete_info = current.pop('snapshot_auto_delete', None)
    self.adjust_sizes(current, after_create)
    if 'type' in self.parameters:
        self.parameters['type'] = self.parameters['type'].lower()
    modify = self.na_helper.get_modified_attributes(current, self.parameters)
    if modify is not None and 'type' in modify:
        msg = 'Error: volume type was not set properly at creation time.' if after_create else 'Error: changing a volume from one type to another is not allowed.'
        msg += '  Current: %s, desired: %s.' % (current['type'], self.parameters['type'])
        self.module.fail_json(msg=msg)
    if modify is not None and 'snaplock' in modify:
        self.validate_snaplock_changes(current, modify, after_create)
    desired_style = self.get_volume_style(None)
    if desired_style is not None and desired_style != self.volume_style:
        msg = 'Error: volume backend was not set properly at creation time.' if after_create else 'Error: changing a volume from one backend to another is not allowed.'
        msg += '  Current: %s, desired: %s.' % (self.volume_style, desired_style)
        self.module.fail_json(msg=msg)
    desired_tcontrol = self.na_helper.safe_get(self.parameters, ['nas_application_template', 'tiering', 'control'])
    if desired_tcontrol in ('required', 'disallowed'):
        warn_or_fail = netapp_utils.get_feature(self.module, 'warn_or_fail_on_fabricpool_backend_change')
        if warn_or_fail in ('warn', 'fail'):
            current_tcontrol = self.tiering_control(current)
            if desired_tcontrol != current_tcontrol:
                msg = 'Error: volume tiering control was not set properly at creation time.' if after_create else 'Error: changing a volume from one backend to another is not allowed.'
                msg += '  Current tiering control: %s, desired: %s.' % (current_tcontrol, desired_tcontrol)
                if warn_or_fail == 'fail':
                    self.module.fail_json(msg=msg)
                self.module.warn('Ignored ' + msg)
        elif warn_or_fail not in (None, 'ignore'):
            self.module.warn("Unexpected value '%s' for warn_or_fail_on_fabricpool_backend_change, expecting: None, 'ignore', 'fail', 'warn'" % warn_or_fail)
    if self.parameters.get('snapshot_auto_delete') is not None:
        auto_delete_modify = self.na_helper.get_modified_attributes(auto_delete_info, self.parameters['snapshot_auto_delete'])
        if len(auto_delete_modify) > 0:
            modify['snapshot_auto_delete'] = auto_delete_modify
    return modify