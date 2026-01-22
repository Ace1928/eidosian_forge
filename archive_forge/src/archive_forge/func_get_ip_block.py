from __future__ import absolute_import, division, print_function
def get_ip_block(ucs, pool_dn, first_addr, last_addr, ip_version):
    if ip_version == 'v6':
        dn_type = '/v6block-'
    else:
        dn_type = '/block-'
    block_dn = pool_dn + dn_type + first_addr + '-' + last_addr
    return ucs.login_handle.query_dn(block_dn)