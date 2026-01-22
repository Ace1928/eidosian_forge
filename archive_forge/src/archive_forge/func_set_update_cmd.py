from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
def set_update_cmd(self):
    """set update command"""
    if not self.changed:
        return
    if self.aftype == 'v4':
        maskstr = self._convertlentomask_(self.mask)
    else:
        maskstr = self.mask
    static_bfd_flag = True
    if self.bfd_session_name:
        static_bfd_flag = False
    if self.next_hop is None:
        next_hop = ''
    else:
        next_hop = self.next_hop
    if self.vrf == '_public_':
        vrf = ''
    else:
        vrf = self.vrf
    if self.destvrf == '_public_':
        destvrf = ''
    else:
        destvrf = self.destvrf
    if self.nhp_interface == 'Invalid0':
        nhp_interface = ''
    else:
        nhp_interface = self.nhp_interface
    if self.state == 'present':
        if self.vrf != '_public_':
            if self.destvrf != '_public_':
                self.updates_cmd.append('ip route-static vpn-instance %s %s %s vpn-instance %s %s' % (vrf, self.prefix, maskstr, destvrf, next_hop))
            else:
                self.updates_cmd.append('ip route-static vpn-instance %s %s %s %s %s' % (vrf, self.prefix, maskstr, nhp_interface, next_hop))
        elif self.destvrf != '_public_':
            self.updates_cmd.append('ip route-static %s %s vpn-instance %s %s' % (self.prefix, maskstr, self.destvrf, next_hop))
        else:
            self.updates_cmd.append('ip route-static %s %s %s %s' % (self.prefix, maskstr, nhp_interface, next_hop))
        if self.pref != 60:
            self.updates_cmd.append(' preference %s' % self.pref)
        if self.tag:
            self.updates_cmd.append(' tag %s' % self.tag)
        if not static_bfd_flag:
            self.updates_cmd.append(' track bfd-session %s' % self.bfd_session_name)
        else:
            self.updates_cmd.append(' bfd enable')
        if self.description:
            self.updates_cmd.append(' description %s' % self.description)
    if self.state == 'absent':
        if self.vrf != '_public_':
            if self.destvrf != '_public_':
                self.updates_cmd.append('undo ip route-static vpn-instance %s %s %s vpn-instance %s %s' % (vrf, self.prefix, maskstr, destvrf, next_hop))
            else:
                self.updates_cmd.append('undo ip route-static vpn-instance %s %s %s %s %s' % (vrf, self.prefix, maskstr, nhp_interface, next_hop))
        elif self.destvrf != '_public_':
            self.updates_cmd.append('undo ip route-static %s %s vpn-instance %s %s' % (self.prefix, maskstr, self.destvrf, self.next_hop))
        else:
            self.updates_cmd.append('undo ip route-static %s %s %s %s' % (self.prefix, maskstr, nhp_interface, next_hop))