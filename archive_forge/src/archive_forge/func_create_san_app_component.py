from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_volume
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_san_app_component(self, modify):
    """Create SAN application component"""
    if modify:
        required_options = ['name']
        action = 'modify'
        if 'lun_count' in modify:
            required_options.append('total_size')
    else:
        required_options = ('name', 'total_size')
        action = 'create'
    for option in required_options:
        if self.parameters.get(option) is None:
            self.module.fail_json(msg="Error: '%s' is required to %s a san application." % (option, action))
    application_component = dict(name=self.parameters['name'])
    if not modify:
        application_component['lun_count'] = 1
    for attr in ('igroup_name', 'lun_count', 'storage_service'):
        if not modify or attr in modify:
            value = self.na_helper.safe_get(self.parameters, ['san_application_template', attr])
            if value is not None:
                application_component[attr] = value
    for attr in ('os_type', 'qos_policy_group', 'qos_adaptive_policy_group', 'total_size'):
        if not self.rest_api.meets_rest_minimum_version(True, 9, 8, 0) and attr in ('os_type', 'qos_policy_group', 'qos_adaptive_policy_group'):
            continue
        if not modify or attr in modify:
            value = self.na_helper.safe_get(self.parameters, [attr])
            if value is not None:
                if attr in ('qos_policy_group', 'qos_adaptive_policy_group'):
                    attr = 'qos'
                    value = dict(policy=dict(name=value))
                application_component[attr] = value
    tiering = self.na_helper.safe_get(self.parameters, ['san_application_template', 'tiering'])
    if tiering is not None and (not modify):
        application_component['tiering'] = {}
        for attr in ('control', 'policy', 'object_stores'):
            value = tiering.get(attr)
            if attr == 'object_stores' and value is not None:
                value = [dict(name=x) for x in value]
            if value is not None:
                application_component['tiering'][attr] = value
    return application_component