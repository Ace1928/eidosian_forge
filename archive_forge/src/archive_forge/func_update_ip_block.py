from __future__ import absolute_import, division, print_function
def update_ip_block(ucs, mo, ip_block, ip_version):
    remove_ip_block(ucs, mo.dn, ip_block, ip_version)
    if not ip_block['state'] == 'absent':
        if ip_version == 'v6':
            from ucsmsdk.mometa.ippool.IppoolIpV6Block import IppoolIpV6Block
            IppoolIpV6Block(parent_mo_or_dn=mo, to=ip_block['ipv6_last_addr'], r_from=ip_block['ipv6_first_addr'], prefix=ip_block['ipv6_prefix'], def_gw=ip_block['ipv6_default_gw'], prim_dns=ip_block['ipv6_primary_dns'], sec_dns=ip_block['ipv6_secondary_dns'])
            ucs.login_handle.add_mo(mo, True)
            ucs.login_handle.commit()
        else:
            from ucsmsdk.mometa.ippool.IppoolBlock import IppoolBlock
            IppoolBlock(parent_mo_or_dn=mo, to=ip_block['last_addr'], r_from=ip_block['first_addr'], subnet=ip_block['subnet_mask'], def_gw=ip_block['default_gw'], prim_dns=ip_block['primary_dns'], sec_dns=ip_block['secondary_dns'])
            ucs.login_handle.add_mo(mo, True)
            ucs.login_handle.commit()