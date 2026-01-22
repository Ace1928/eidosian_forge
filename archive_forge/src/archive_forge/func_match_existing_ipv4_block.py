from __future__ import absolute_import, division, print_function
def match_existing_ipv4_block(ucs, dn, ipv4_block):
    mo_1 = get_ip_block(ucs, dn, ipv4_block['first_addr'], ipv4_block['last_addr'], 'v4')
    if not mo_1:
        if ipv4_block['state'] == 'absent':
            return True
        return False
    else:
        if ipv4_block['state'] == 'absent':
            return False
        kwargs = dict(subnet=ipv4_block['subnet_mask'])
        kwargs['def_gw'] = ipv4_block['default_gw']
        kwargs['prim_dns'] = ipv4_block['primary_dns']
        kwargs['sec_dns'] = ipv4_block['secondary_dns']
        return mo_1.check_prop_match(**kwargs)