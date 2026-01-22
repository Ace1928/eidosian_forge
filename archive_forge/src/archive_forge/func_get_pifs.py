from __future__ import absolute_import, division, print_function
from ansible.module_utils import distro
from ansible.module_utils.basic import AnsibleModule
def get_pifs(session):
    recs = session.xenapi.PIF.get_all_records()
    pifs = change_keys(recs, key='uuid')
    xs_pifs = {}
    devicenums = range(0, 7)
    for pif in pifs.values():
        for eth in devicenums:
            interface_name = 'eth%s' % eth
            bond_name = interface_name.replace('eth', 'bond')
            if pif['device'] == interface_name:
                xs_pifs[interface_name] = pif
            elif pif['device'] == bond_name:
                xs_pifs[bond_name] = pif
    return xs_pifs