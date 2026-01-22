import argparse
from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.vpn import utils as vpn_utils
def parse_common_args2body(parsed_args, body):
    neutronv20.update_dict(parsed_args, body, ['auth_algorithm', 'encryption_algorithm', 'encapsulation_mode', 'transform_protocol', 'pfs', 'name', 'description', 'tenant_id'])
    if parsed_args.lifetime:
        vpn_utils.validate_lifetime_dict(parsed_args.lifetime)
        body['lifetime'] = parsed_args.lifetime
    return body