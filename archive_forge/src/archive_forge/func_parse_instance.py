from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils._text import to_bytes
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import string_types
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.argspec.routing_instances.routing_instances import (
def parse_instance(self, instance):
    """

        :param instance:
        :return:
        """
    instance_dict = {}
    instance_dict['name'] = instance['name']
    if 'connector-id-advertise' in instance.keys():
        instance_dict['connector_id_advertise'] = True
    if instance.get('description'):
        instance_dict['description'] = instance['description']
    if instance.get('instance-role'):
        instance_dict['instance_role'] = instance['instance-role']
    if instance.get('instance-type'):
        instance_dict['type'] = instance['instance-type']
    if instance.get('interface'):
        interfaces = instance.get('interface')
        interfaces_list = []
        if isinstance(interfaces, list):
            for interface in interfaces:
                interfaces_list.append(self.parse_interface(interface))
        else:
            interfaces_list.append(self.parse_interface(interfaces))
        instance_dict['interfaces'] = interfaces_list
    if instance.get('l2vpn-id'):
        instance_dict['l2vpn_id'] = instance['l2vpn-id'].get('community')
    if 'no-irb-layer2-copy' in instance.keys():
        instance_dict['no_irb_layer_2_copy'] = True
    if 'no-local-switching' in instance.keys():
        instance_dict['no_local_switching'] = True
    if 'no-vrf-advertise' in instance.keys():
        instance_dict['no_vrf_advertise'] = True
    if 'no-vrf-propagate-ttl' in instance.keys():
        instance_dict['no_vrf_propagate_ttl'] = True
    if instance.get('qualified-bum-pruning-mode'):
        instance_dict['qualified_bum_pruning_mode'] = True
    if instance.get('route-distinguisher'):
        instance_dict['route_distinguisher'] = instance['route-distinguisher'].get('rd-type')
    if instance.get('vrf-import'):
        vrf_imp_lst = []
        vrf_imp = instance.get('vrf-import')
        if isinstance(vrf_imp, list):
            vrf_imp_lst = vrf_imp
        else:
            vrf_imp_lst.append(vrf_imp)
        instance_dict['vrf_imports'] = vrf_imp_lst
    if instance.get('vrf-export'):
        vrf_exp_lst = []
        vrf_exp = instance.get('vrf-export')
        if isinstance(vrf_exp, list):
            vrf_exp_lst = vrf_exp
        else:
            vrf_exp_lst.append(vrf_exp)
        instance_dict['vrf_exports'] = vrf_exp_lst
    return utils.remove_empties(instance_dict)