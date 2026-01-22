from __future__ import absolute_import, division, print_function
import base64
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def deploy_instance(self, start_vm=True):
    self.result['changed'] = True
    networkids = self.get_network_ids()
    if networkids is not None:
        networkids = ','.join(networkids)
    args = {}
    args['templateid'] = self.get_template_or_iso(key='id')
    if not args['templateid']:
        self.module.fail_json(msg='Template or ISO is required.')
    args['zoneid'] = self.get_zone(key='id')
    args['serviceofferingid'] = self.get_service_offering_id()
    args['account'] = self.get_account(key='name')
    args['domainid'] = self.get_domain(key='id')
    args['projectid'] = self.get_project(key='id')
    args['diskofferingid'] = self.get_disk_offering(key='id')
    args['networkids'] = networkids
    args['iptonetworklist'] = self.get_iptonetwork_mappings()
    args['userdata'] = self.get_user_data()
    args['keyboard'] = self.module.params.get('keyboard')
    args['ipaddress'] = self.module.params.get('ip_address')
    args['ip6address'] = self.module.params.get('ip6_address')
    args['name'] = self.module.params.get('name')
    args['displayname'] = self.get_or_fallback('display_name', 'name')
    args['group'] = self.module.params.get('group')
    args['keypair'] = self.get_ssh_keypair(key='name')
    args['size'] = self.module.params.get('disk_size')
    args['startvm'] = start_vm
    args['rootdisksize'] = self.module.params.get('root_disk_size')
    args['affinitygroupnames'] = self.module.params.get('affinity_groups')
    args['details'] = self.get_details()
    args['securitygroupnames'] = self.module.params.get('security_groups')
    args['hostid'] = self.get_host_id()
    args['clusterid'] = self.get_cluster_id()
    args['podid'] = self.get_pod_id()
    template_iso = self.get_template_or_iso()
    if 'hypervisor' not in template_iso:
        args['hypervisor'] = self.get_hypervisor()
    instance = None
    if not self.module.check_mode:
        instance = self.query_api('deployVirtualMachine', **args)
        poll_async = self.module.params.get('poll_async')
        if poll_async:
            instance = self.poll_job(instance, 'virtualmachine')
    return instance