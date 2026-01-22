import struct
import sys
def collapse_address_list(addresses):
    """Collapse a list of IP objects.

    Example:
        collapse_address_list([IPv4('1.1.0.0/24'), IPv4('1.1.1.0/24')]) ->
          [IPv4('1.1.0.0/23')]

    Args:
        addresses: A list of IPv4Network or IPv6Network objects.

    Returns:
        A list of IPv4Network or IPv6Network objects depending on what we
        were passed.

    Raises:
        TypeError: If passed a list of mixed version objects.

    """
    i = 0
    addrs = []
    ips = []
    nets = []
    for ip in addresses:
        if isinstance(ip, _BaseIP):
            if ips and ips[-1]._version != ip._version:
                raise TypeError('%s and %s are not of the same version' % (str(ip), str(ips[-1])))
            ips.append(ip)
        elif ip._prefixlen == ip._max_prefixlen:
            if ips and ips[-1]._version != ip._version:
                raise TypeError('%s and %s are not of the same version' % (str(ip), str(ips[-1])))
            ips.append(ip.ip)
        else:
            if nets and nets[-1]._version != ip._version:
                raise TypeError('%s and %s are not of the same version' % (str(ip), str(nets[-1])))
            nets.append(ip)
    ips = sorted(set(ips))
    nets = sorted(set(nets))
    while i < len(ips):
        first, last, last_index = _find_address_range(ips[i:])
        i += last_index + 1
        addrs.extend(summarize_address_range(first, last))
    return _collapse_address_list_recursive(sorted(addrs + nets, key=_BaseNet._get_networks_key))