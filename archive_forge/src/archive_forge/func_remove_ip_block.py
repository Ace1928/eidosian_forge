from __future__ import absolute_import, division, print_function
def remove_ip_block(ucs, dn, ip_block, ip_version):
    if ip_version == 'v6':
        first_addr = ip_block['ipv6_first_addr']
        last_addr = ip_block['ipv6_last_addr']
    else:
        first_addr = ip_block['first_addr']
        last_addr = ip_block['last_addr']
    mo_1 = get_ip_block(ucs, dn, first_addr, last_addr, ip_version)
    if mo_1:
        ucs.login_handle.remove_mo(mo_1)
        ucs.login_handle.commit()