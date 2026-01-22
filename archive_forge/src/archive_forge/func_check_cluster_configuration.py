from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils._text import to_native
def check_cluster_configuration(self):
    """
        Check cluster configuration
        Returns: 'Present' if cluster exists, else 'absent'

        """
    try:
        self.datacenter = find_datacenter_by_name(self.content, self.datacenter_name)
        if self.datacenter is None:
            self.module.fail_json(msg='Datacenter %s does not exist.' % self.datacenter_name)
        self.cluster = self.find_cluster_by_name(cluster_name=self.cluster_name, datacenter_name=self.datacenter)
        if self.cluster is None:
            return 'absent'
        return 'present'
    except vmodl.RuntimeFault as runtime_fault:
        self.module.fail_json(msg=to_native(runtime_fault.msg))
    except vmodl.MethodFault as method_fault:
        self.module.fail_json(msg=to_native(method_fault.msg))
    except Exception as generic_exc:
        self.module.fail_json(msg='Failed to check configuration due to generic exception %s' % to_native(generic_exc))