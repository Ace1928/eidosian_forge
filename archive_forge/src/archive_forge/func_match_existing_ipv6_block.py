from __future__ import absolute_import, division, print_function
def match_existing_ipv6_block(ucs, dn, ipv6_block):
    mo_1 = get_ip_block(ucs, dn, ipv6_block['ipv6_first_addr'], ipv6_block['ipv6_last_addr'], 'v6')
    if not mo_1:
        if ipv6_block['state'] == 'absent':
            return True
        return False
    else:
        if ipv6_block['state'] == 'absent':
            return False
        kwargs = dict(prefix=ipv6_block['ipv6_prefix'])
        kwargs['def_gw'] = ipv6_block['ipv6_default_gw']
        kwargs['prim_dns'] = ipv6_block['ipv6_primary_dns']
        kwargs['sec_dns'] = ipv6_block['ipv6_secondary_dns']
        return mo_1.check_prop_match(**kwargs)