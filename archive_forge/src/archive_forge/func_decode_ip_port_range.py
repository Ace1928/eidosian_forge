import json
import netaddr
import re
def decode_ip_port_range(value):
    """
    Decodes an IP and port range:
        {ip_start}-{ip-end}:{port_start}-{port_end}

    IPv6 addresses are surrounded by "[" and "]" if port ranges are also
    present

    Returns the following dictionary:
        {
            "addrs": {
                "start": {ip_start}
                "end": {ip_end}
            }
            "ports": {
                "start": {port_start},
                "end": {port_end}
        }
        (the "ports" key might be omitted)
    """
    if value.count(':') > 1:
        match = ipv6_port_regex.match(value)
    else:
        match = ipv4_port_regex.match(value)
    ip_start = match.group(1)
    ip_end = match.group(2)
    port_start = match.group(3)
    port_end = match.group(4)
    result = {'addrs': {'start': netaddr.IPAddress(ip_start), 'end': netaddr.IPAddress(ip_end or ip_start)}}
    if port_start:
        result['ports'] = {'start': int(port_start), 'end': int(port_end or port_start)}
    return result