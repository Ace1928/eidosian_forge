from osc_lib import exceptions
from openstackclient.i18n import _
def is_ipv6_protocol(protocol):
    if protocol is not None and protocol.startswith('ipv6-') or protocol in ['icmpv6', '41', '43', '44', '58', '59', '60']:
        return True
    else:
        return False