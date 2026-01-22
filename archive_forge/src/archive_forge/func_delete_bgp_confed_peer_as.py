from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def delete_bgp_confed_peer_as(self, **kwargs):
    """ delete_bgp_confed_peer_as """
    module = kwargs['module']
    confed_peer_as_num = module.params['confed_peer_as_num']
    conf_str = CE_DELETE_BGP_CONFED_PEER_AS % confed_peer_as_num
    recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
    if '<ok/>' not in recv_xml:
        module.fail_json(msg='Error: Delete bgp confed peer as failed.')
    cmds = []
    cmd = 'undo confederation peer-as %s' % confed_peer_as_num
    cmds.append(cmd)
    return cmds