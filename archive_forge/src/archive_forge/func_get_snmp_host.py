from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_snmp_host(host, udp, module):
    body = execute_show_command('show snmp host', module)
    host_map = {'port': 'udp', 'version': 'version', 'level': 'v3', 'type': 'snmp_type', 'secname': 'community'}
    host_map_5k = {'port': 'udp', 'version': 'version', 'sec_level': 'v3', 'notif_type': 'snmp_type', 'commun_or_user': 'community'}
    resource = {}
    if body:
        try:
            resource_table = body[0]['TABLE_host']['ROW_host']
            if isinstance(resource_table, dict):
                resource_table = [resource_table]
            for each in resource_table:
                key = str(each['host']) + '_' + str(each['port']).strip()
                src = each.get('src_intf')
                host_resource = apply_key_map(host_map, each)
                if src:
                    host_resource['src_intf'] = src
                    if re.search('interface:', src):
                        host_resource['src_intf'] = src.split(':')[1].strip()
                vrf_filt = each.get('TABLE_vrf_filters')
                if vrf_filt:
                    vrf_filter = vrf_filt['ROW_vrf_filters']['vrf_filter'].split(':')[1].split(',')
                    filters = [vrf.strip() for vrf in vrf_filter]
                    host_resource['vrf_filter'] = filters
                vrf = each.get('vrf')
                if vrf:
                    host_resource['vrf'] = vrf.split(':')[1].strip()
                resource[key] = host_resource
        except KeyError:
            try:
                resource_table = body[0]['TABLE_hosts']['ROW_hosts']
                if isinstance(resource_table, dict):
                    resource_table = [resource_table]
                for each in resource_table:
                    key = str(each['address']) + '_' + str(each['port']).strip()
                    src = each.get('src_intf')
                    host_resource = apply_key_map(host_map_5k, each)
                    if src:
                        host_resource['src_intf'] = src
                        if re.search('interface:', src):
                            host_resource['src_intf'] = src.split(':')[1].strip()
                    vrf = each.get('use_vrf_name')
                    if vrf:
                        host_resource['vrf'] = vrf.strip()
                    vrf_filt = each.get('TABLE_filter_vrf')
                    if vrf_filt:
                        vrf_filter = vrf_filt['ROW_filter_vrf']['filter_vrf_name'].split(',')
                        filters = [vrf.strip() for vrf in vrf_filter]
                        host_resource['vrf_filter'] = filters
                    resource[key] = host_resource
            except (KeyError, AttributeError, TypeError):
                return resource
        except (AttributeError, TypeError):
            return resource
        find = resource.get(host + '_' + udp)
        if find:
            fix_find = {}
            for key, value in find.items():
                if isinstance(value, str):
                    fix_find[key] = value.strip()
                else:
                    fix_find[key] = value
            return fix_find
    return {}