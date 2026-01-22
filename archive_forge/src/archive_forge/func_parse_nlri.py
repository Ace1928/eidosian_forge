from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils._text import to_bytes
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import string_types
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.argspec.bgp_address_family.bgp_address_family import (
def parse_nlri(self, cfg, nlri_t):
    """

        :param cfg:
        :return:
        """
    nlri_dict = {}
    if cfg and nlri_t in cfg.keys():
        nlri_dict['type'] = nlri_t
        nlri = cfg.get(nlri_t)
        if not nlri:
            nlri_dict['set'] = True
            return nlri_dict
        if 'accepted-prefix-limit' in nlri.keys():
            apl_dict = self.parse_accepted_prefix_limit(nlri)
            if apl_dict:
                nlri_dict['accepted_prefix_limit'] = apl_dict
        if 'add-path' in nlri.keys():
            ap_dict = self.parse_add_path(nlri)
            if ap_dict:
                nlri_dict['add_path'] = ap_dict
        if 'aggregate-label' in nlri.keys():
            al_dict = self.parse_aggregate_label(nlri)
            if apl_dict:
                nlri_dict['aggregate_label'] = al_dict
        if 'aigp' in nlri.keys():
            aigp_dict = self.parse_aigp(nlri)
            if aigp_dict:
                nlri_dict['aigp'] = aigp_dict
        if 'damping' in nlri.keys():
            nlri_dict['damping'] = True
        if 'defer-initial-multipath-build' in nlri.keys():
            dimb_dict = self.parse_defer_initial_multipath_build(nlri)
            if dimb_dict:
                nlri_dict['defer_initial_multipath_build'] = dimb_dict
        if 'delay-route-advertisements' in nlri.keys():
            dra_dict = self.parse_delay_route_advertisements(nlri)
            if dra_dict:
                nlri_dict['delay_route_advertisements'] = dra_dict
        if 'entropy-label' in nlri.keys():
            el_dict = self.parse_entropy_label(nlri)
            if el_dict:
                nlri_dict['entropy_label'] = el_dict
        if 'explicit-null' in nlri.keys():
            en_dict = self.parse_explicit_null(nlri)
            if en_dict:
                nlri_dict['explicit_null'] = en_dict
        if 'extended-nexthop' in nlri.keys():
            nlri_dict['extended_nexthop'] = True
        if 'extended-nexthop-color' in nlri.keys():
            nlri_dict['extended_nexthop_color'] = True
        if 'graceful-restart' in nlri.keys():
            gr = nlri.get('graceful-restart')
            if 'forwarding-state-bit' in gr.keys():
                fsb = gr.get('forwarding-state-bit')
                nlri_dict['graceful_restart_forwarding_state_bit'] = fsb
        if 'legacy-redirect-ip-action' in nlri.keys():
            lria_dict = self.parse_legacy_redirect_ip_action(nlri)
            if lria_dict:
                nlri_dict['legacy_redirect_ip_action'] = lria_dict
        if 'local-ipv4-address' in nlri.keys():
            nlri_dict['local_ipv4_address'] = nlri.get('local-ipv4-address')
        if 'loops' in nlri.keys():
            loops = nlri.get('loops')
            nlri_dict['loops'] = loops.get('loops')
        if 'no-install' in nlri.keys():
            nlri_dict['no_install'] = True
        if 'no-validate' in nlri.keys():
            nlri_dict['no_validate'] = nlri.get('no-validate')
        if 'output-queue-priority' in nlri.keys():
            oqp = nlri.get('output-queue-priority')
            if 'expedited' in oqp.keys():
                nlri_dict['output_queue_priority_expedited'] = True
            if 'priority' in oqp.keys():
                nlri_dict['output_queue_priority_priority'] = oqp.get('priority')
        if 'per-group-label' in nlri.keys():
            nlri_dict['per_group_label'] = True
        if 'per-prefix-label' in nlri.keys():
            nlri_dict['per_prefix_label'] = True
        if 'resolve-vpn' in nlri.keys():
            nlri_dict['resolve_vpn'] = True
        if 'prefix-limit' in nlri.keys():
            pl_dict = self.parse_accepted_prefix_limit(nlri)
            if pl_dict:
                nlri_dict['prefix_limit'] = pl_dict
        if 'resolve-vpn' in nlri.keys():
            nlri_dict['resolve_vpn'] = True
        if 'rib' in nlri.keys():
            nlri_dict['rib'] = 'inet.3'
        if 'rib-group' in nlri.keys():
            nlri_dict['rib_group'] = nlri.get('rib-group')
        if 'route-refresh-priority' in nlri.keys():
            oqp = nlri.get('route-refresh-priority')
            if 'expedited' in oqp.keys():
                nlri_dict['route_refresh_priority_expedited'] = True
            if 'priority' in oqp.keys():
                nlri_dict['route_refresh_priority_priority'] = oqp.get('priority')
        if 'secondary-independent-resolution' in nlri.keys():
            nlri_dict['secondary_independent_resolution'] = True
        if 'strip-nexthop' in nlri.keys():
            nlri_dict['strip_nexthop'] = True
        if 'topology' in nlri.keys():
            t_list = self.parse_topology(nlri)
            if t_list:
                nlri_dict['topology'] = t_list
        if 'traffic-statistics' in nlri.keys():
            ts_dict = self.parse_traffic_statistics(nlri)
            if ts_dict:
                nlri_dict['traffic_statistics'] = ts_dict
        if 'withdraw-priority' in nlri.keys():
            oqp = nlri.get('withdraw-priority')
            if 'expedited' in oqp.keys():
                nlri_dict['withdraw_priority_expedited'] = True
            if 'priority' in oqp.keys():
                nlri_dict['withdraw_priority_priority'] = oqp.get('priority')
        return nlri_dict