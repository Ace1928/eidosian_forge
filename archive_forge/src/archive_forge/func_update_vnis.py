from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.bgp_af.bgp_af import Bgp_afArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
def update_vnis(self, data):
    for conf in data:
        afs = conf.get('address_family', [])
        if afs:
            for af in afs:
                vnis = af.get('vnis', None)
                if vnis:
                    vnis_list = []
                    for vni in vnis:
                        vni_dict = {}
                        vni_config = vni['config']
                        vni_number = vni_config.get('vni-number', None)
                        vni_adv_gw = vni_config.get('advertise-default-gw', None)
                        vni_adv_svi = vni_config.get('advertise-svi-ip', None)
                        vni_rd = vni_config.get('route-distinguisher', None)
                        vni_rt_in = vni_config.get('import-rts', [])
                        vni_rt_out = vni_config.get('export-rts', [])
                        if vni_number:
                            vni_dict['vni_number'] = vni_number
                        if vni_adv_gw is not None:
                            vni_dict['advertise_default_gw'] = vni_adv_gw
                        if vni_adv_svi is not None:
                            vni_dict['advertise_svi_ip'] = vni_adv_svi
                        if vni_rd:
                            vni_dict['rd'] = vni_rd
                        if vni_rt_in:
                            vni_dict['rt_in'] = vni_rt_in
                        if vni_rt_out:
                            vni_dict['rt_out'] = vni_rt_out
                        vnis_list.append(vni_dict)
                    af['vnis'] = vnis_list