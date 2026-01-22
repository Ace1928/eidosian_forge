from osc_lib import exceptions
from openstackclient.i18n import _
def format_remote_ip_prefix(rule):
    remote_ip_prefix = rule['remote_ip_prefix']
    if remote_ip_prefix is None:
        ethertype = rule['ether_type']
        if ethertype == 'IPv4':
            remote_ip_prefix = '0.0.0.0/0'
        elif ethertype == 'IPv6':
            remote_ip_prefix = '::/0'
    return remote_ip_prefix