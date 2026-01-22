from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.qos import rule as qos_rule
def update_bandwidth_limit_args2body(parsed_args, body):
    max_kbps = parsed_args.max_kbps
    max_burst_kbps = parsed_args.max_burst_kbps
    if not (max_kbps or max_burst_kbps):
        raise exceptions.CommandError(_('Must provide max-kbps or max-burst-kbps option.'))
    neutronv20.update_dict(parsed_args, body, ['max_kbps', 'max_burst_kbps', 'tenant_id'])