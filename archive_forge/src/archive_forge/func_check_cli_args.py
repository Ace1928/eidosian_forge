from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, \
def check_cli_args(self):
    """ Check invalid cli args """
    if self.connect_port:
        if int(self.connect_port) != 161 and (int(self.connect_port) > 65535 or int(self.connect_port) < 1025):
            self.module.fail_json(msg='Error: The value of connect_port %s is out of [161, 1025 - 65535].' % self.connect_port)