from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_text
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
def get_net_stack_instance_config(self):
    """
        Get a configuration of tcpip stack item if it is enabled.
        """
    self.exist_net_stack_instance_config = {}
    for key, value in self.enabled_net_stack_instance.items():
        if value is True:
            for net_stack_instance in self.host_obj.config.network.netStackInstance:
                if net_stack_instance.key == self.net_stack_instance_keys[key]:
                    self.exist_net_stack_instance_config[key] = net_stack_instance