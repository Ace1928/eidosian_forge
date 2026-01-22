from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from distutils.version import LooseVersion
def gluster_peer_ops(self):
    if not self.nodes:
        self.module.fail_json(msg='nodes list cannot be empty')
    self.force = 'force' if self.module.params.get('force') else ''
    if self.state == 'present':
        self.nodes = self.get_to_be_probed_hosts(self.nodes)
        self.action = 'probe'
        self.force = ''
    else:
        self.action = 'detach'
    self.call_peer_commands()