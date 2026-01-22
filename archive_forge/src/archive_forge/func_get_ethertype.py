from osc_lib import exceptions
from openstackclient.i18n import _
def get_ethertype(parsed_args, protocol):
    ethertype = 'IPv4'
    if parsed_args.ethertype is not None:
        ethertype = parsed_args.ethertype
    elif is_ipv6_protocol(protocol):
        ethertype = 'IPv6'
    return ethertype