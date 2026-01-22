from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
def check_listener_attrs(attrs):
    if 'protocol_port' in attrs:
        _validate_TCP_UDP_SCTP_port_range(attrs['protocol_port'], 'protocol-port')
    extra_hsts_opts_set = attrs.get('hsts_preload') or attrs.get('hsts_include_subdomains')
    if extra_hsts_opts_set and 'hsts_max_age' not in attrs:
        raise exceptions.InvalidValue('Argument hsts_max_age is required when using hsts_preload or hsts_include_subdomains arguments.')