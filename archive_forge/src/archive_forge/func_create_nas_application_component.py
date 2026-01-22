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
def create_nas_application_component(self):
    """Create application component for nas template"""
    required_options = ('name', 'size')
    for option in required_options:
        if self.parameters.get(option) is None:
            self.module.fail_json(msg='Error: "%s" is required to create nas application.' % option)
    application_component = dict(name=self.parameters['name'], total_size=self.parameters['size'], share_count=1, scale_out=self.volume_style == 'flexgroup')
    name = self.na_helper.safe_get(self.parameters, ['nas_application_template', 'storage_service'])
    if name is not None:
        application_component['storage_service'] = dict(name=name)
    flexcache = self.na_helper.safe_get(self.parameters, ['nas_application_template', 'flexcache'])
    if flexcache is not None:
        application_component['flexcache'] = dict(origin=dict(svm=dict(name=flexcache['origin_svm_name']), component=dict(name=flexcache['origin_component_name'])))
        del application_component['scale_out']
        dr_cache = self.na_helper.safe_get(self.parameters, ['nas_application_template', 'flexcache', 'dr_cache'])
        if dr_cache is not None:
            application_component['flexcache']['dr_cache'] = dr_cache
    tiering = self.na_helper.safe_get(self.parameters, ['nas_application_template', 'tiering'])
    if tiering is not None or self.parameters.get('tiering_policy') is not None:
        application_component['tiering'] = {}
        if tiering is None:
            tiering = {}
        if 'policy' not in tiering:
            tiering['policy'] = self.parameters.get('tiering_policy')
        for attr in ('control', 'policy', 'object_stores'):
            value = tiering.get(attr)
            if attr == 'object_stores' and value is not None:
                value = [dict(name=x) for x in value]
            if value is not None:
                application_component['tiering'][attr] = value
    if self.get_qos_policy_group() is not None:
        application_component['qos'] = {'policy': {'name': self.get_qos_policy_group()}}
    if self.parameters.get('export_policy') is not None:
        application_component['export_policy'] = {'name': self.parameters['export_policy']}
    return application_component