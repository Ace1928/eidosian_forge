from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
def get_datacenter_info(self):
    self.datacenter_name = self.params.get('datacenter')
    results = dict(changed=False, datacenter_info=[])
    datacenter_objs = self.get_managed_objects_properties(vim_type=vim.Datacenter, properties=['name'])
    dcs = []
    for dc_obj in datacenter_objs:
        if len(dc_obj.propSet) == 1:
            if self.datacenter_name is not None:
                if dc_obj.propSet[0].val == to_native(self.datacenter_name):
                    dcs.append(dc_obj.obj)
                    continue
            else:
                dcs.append(dc_obj.obj)
    for obj in dcs:
        if obj is None:
            continue
        temp_dc = dict(name=obj.name, moid=obj._moId)
        if self.module.params['schema'] == 'summary':
            temp_dc.update(dict(config_status=obj.configStatus, overall_status=obj.overallStatus))
        else:
            temp_dc.update(self.to_json(obj, self.params.get('properties')))
        if self.params.get('show_tag'):
            temp_dc.update({'tags': self.vmware_client.get_tags_for_datacenter(datacenter_mid=obj._moId)})
        results['datacenter_info'].append(temp_dc)
    self.module.exit_json(**results)