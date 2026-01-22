from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def check_bgp_enable_other_args(self, **kwargs):
    """ check_bgp_enable_other_args """
    module = kwargs['module']
    state = module.params['state']
    result = dict()
    need_cfg = False
    graceful_restart = module.params['graceful_restart']
    if graceful_restart != 'no_use':
        conf_str = CE_GET_BGP_ENABLE_HEADER + '<gracefulRestart></gracefulRestart>' + CE_GET_BGP_ENABLE_TAIL
        recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
        if '<data/>' in recv_xml:
            need_cfg = True
        else:
            re_find = re.findall('.*<gracefulRestart>(.*)</gracefulRestart>.*', recv_xml)
            if re_find:
                result['graceful_restart'] = re_find
                if re_find[0] != graceful_restart:
                    need_cfg = True
            else:
                need_cfg = True
    time_wait_for_rib = module.params['time_wait_for_rib']
    if time_wait_for_rib:
        if int(time_wait_for_rib) > 3000 or int(time_wait_for_rib) < 3:
            module.fail_json(msg='Error: The time_wait_for_rib %s is out of [3 - 3000].' % time_wait_for_rib)
        else:
            conf_str = CE_GET_BGP_ENABLE_HEADER + '<timeWaitForRib></timeWaitForRib>' + CE_GET_BGP_ENABLE_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if state == 'present':
                if '<data/>' in recv_xml:
                    need_cfg = True
                else:
                    re_find = re.findall('.*<timeWaitForRib>(.*)</timeWaitForRib>.*', recv_xml)
                    if re_find:
                        result['time_wait_for_rib'] = re_find
                        if re_find[0] != time_wait_for_rib:
                            need_cfg = True
                    else:
                        need_cfg = True
            elif '<data/>' in recv_xml:
                pass
            else:
                re_find = re.findall('.*<timeWaitForRib>(.*)</timeWaitForRib>.*', recv_xml)
                if re_find:
                    result['time_wait_for_rib'] = re_find
                    if re_find[0] == time_wait_for_rib:
                        need_cfg = True
    as_path_limit = module.params['as_path_limit']
    if as_path_limit:
        if int(as_path_limit) > 2000 or int(as_path_limit) < 1:
            module.fail_json(msg='Error: The as_path_limit %s is out of [1 - 2000].' % as_path_limit)
        else:
            conf_str = CE_GET_BGP_ENABLE_HEADER + '<asPathLimit></asPathLimit>' + CE_GET_BGP_ENABLE_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if state == 'present':
                if '<data/>' in recv_xml:
                    need_cfg = True
                else:
                    re_find = re.findall('.*<asPathLimit>(.*)</asPathLimit>.*', recv_xml)
                    if re_find:
                        result['as_path_limit'] = re_find
                        if re_find[0] != as_path_limit:
                            need_cfg = True
                    else:
                        need_cfg = True
            elif '<data/>' in recv_xml:
                pass
            else:
                re_find = re.findall('.*<asPathLimit>(.*)</asPathLimit>.*', recv_xml)
                if re_find:
                    result['as_path_limit'] = re_find
                    if re_find[0] == as_path_limit:
                        need_cfg = True
    check_first_as = module.params['check_first_as']
    if check_first_as != 'no_use':
        conf_str = CE_GET_BGP_ENABLE_HEADER + '<checkFirstAs></checkFirstAs>' + CE_GET_BGP_ENABLE_TAIL
        recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
        if '<data/>' in recv_xml:
            need_cfg = True
        else:
            re_find = re.findall('.*<checkFirstAs>(.*)</checkFirstAs>.*', recv_xml)
            if re_find:
                result['check_first_as'] = re_find
                if re_find[0] != check_first_as:
                    need_cfg = True
            else:
                need_cfg = True
    confed_id_number = module.params['confed_id_number']
    if confed_id_number:
        if len(confed_id_number) > 11 or len(confed_id_number) == 0:
            module.fail_json(msg='Error: The len of confed_id_number %s is out of [1 - 11].' % confed_id_number)
        else:
            conf_str = CE_GET_BGP_ENABLE_HEADER + '<confedIdNumber></confedIdNumber>' + CE_GET_BGP_ENABLE_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if state == 'present':
                if '<data/>' in recv_xml:
                    need_cfg = True
                else:
                    re_find = re.findall('.*<confedIdNumber>(.*)</confedIdNumber>.*', recv_xml)
                    if re_find:
                        result['confed_id_number'] = re_find
                        if re_find[0] != confed_id_number:
                            need_cfg = True
                    else:
                        need_cfg = True
            elif '<data/>' in recv_xml:
                pass
            else:
                re_find = re.findall('.*<confedIdNumber>(.*)</confedIdNumber>.*', recv_xml)
                if re_find:
                    result['confed_id_number'] = re_find
                    if re_find[0] == confed_id_number:
                        need_cfg = True
    confed_nonstanded = module.params['confed_nonstanded']
    if confed_nonstanded != 'no_use':
        conf_str = CE_GET_BGP_ENABLE_HEADER + '<confedNonstanded></confedNonstanded>' + CE_GET_BGP_ENABLE_TAIL
        recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
        if '<data/>' in recv_xml:
            need_cfg = True
        else:
            re_find = re.findall('.*<confedNonstanded>(.*)</confedNonstanded>.*', recv_xml)
            if re_find:
                result['confed_nonstanded'] = re_find
                if re_find[0] != confed_nonstanded:
                    need_cfg = True
            else:
                need_cfg = True
    bgp_rid_auto_sel = module.params['bgp_rid_auto_sel']
    if bgp_rid_auto_sel != 'no_use':
        conf_str = CE_GET_BGP_ENABLE_HEADER + '<bgpRidAutoSel></bgpRidAutoSel>' + CE_GET_BGP_ENABLE_TAIL
        recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
        if '<data/>' in recv_xml:
            need_cfg = True
        else:
            re_find = re.findall('.*<bgpRidAutoSel>(.*)</bgpRidAutoSel>.*', recv_xml)
            if re_find:
                result['bgp_rid_auto_sel'] = re_find
                if re_find[0] != bgp_rid_auto_sel:
                    need_cfg = True
            else:
                need_cfg = True
    keep_all_routes = module.params['keep_all_routes']
    if keep_all_routes != 'no_use':
        conf_str = CE_GET_BGP_ENABLE_HEADER + '<keepAllRoutes></keepAllRoutes>' + CE_GET_BGP_ENABLE_TAIL
        recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
        if '<data/>' in recv_xml:
            need_cfg = True
        else:
            re_find = re.findall('.*<keepAllRoutes>(.*)</keepAllRoutes>.*', recv_xml)
            if re_find:
                result['keep_all_routes'] = re_find
                if re_find[0] != keep_all_routes:
                    need_cfg = True
            else:
                need_cfg = True
    memory_limit = module.params['memory_limit']
    if memory_limit != 'no_use':
        conf_str = CE_GET_BGP_ENABLE_HEADER + '<memoryLimit></memoryLimit>' + CE_GET_BGP_ENABLE_TAIL
        recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
        if '<data/>' in recv_xml:
            need_cfg = True
        else:
            re_find = re.findall('.*<memoryLimit>(.*)</memoryLimit>.*', recv_xml)
            if re_find:
                result['memory_limit'] = re_find
                if re_find[0] != memory_limit:
                    need_cfg = True
            else:
                need_cfg = True
    gr_peer_reset = module.params['gr_peer_reset']
    if gr_peer_reset != 'no_use':
        conf_str = CE_GET_BGP_ENABLE_HEADER + '<grPeerReset></grPeerReset>' + CE_GET_BGP_ENABLE_TAIL
        recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
        if '<data/>' in recv_xml:
            need_cfg = True
        else:
            re_find = re.findall('.*<grPeerReset>(.*)</grPeerReset>.*', recv_xml)
            if re_find:
                result['gr_peer_reset'] = re_find
                if re_find[0] != gr_peer_reset:
                    need_cfg = True
            else:
                need_cfg = True
    is_shutdown = module.params['is_shutdown']
    if is_shutdown != 'no_use':
        conf_str = CE_GET_BGP_ENABLE_HEADER + '<isShutdown></isShutdown>' + CE_GET_BGP_ENABLE_TAIL
        recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
        if '<data/>' in recv_xml:
            need_cfg = True
        else:
            re_find = re.findall('.*<isShutdown>(.*)</isShutdown>.*', recv_xml)
            if re_find:
                result['is_shutdown'] = re_find
                if re_find[0] != is_shutdown:
                    need_cfg = True
            else:
                need_cfg = True
    suppress_interval = module.params['suppress_interval']
    hold_interval = module.params['hold_interval']
    clear_interval = module.params['clear_interval']
    if suppress_interval:
        if not hold_interval or not clear_interval:
            module.fail_json(msg='Error: Please input suppress_interval hold_interval clear_interval at the same time.')
        if int(suppress_interval) > 65535 or int(suppress_interval) < 1:
            module.fail_json(msg='Error: The suppress_interval %s is out of [1 - 65535].' % suppress_interval)
        else:
            conf_str = CE_GET_BGP_ENABLE_HEADER + '<suppressInterval></suppressInterval>' + CE_GET_BGP_ENABLE_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if state == 'present':
                if '<data/>' in recv_xml:
                    need_cfg = True
                else:
                    re_find = re.findall('.*<suppressInterval>(.*)</suppressInterval>.*', recv_xml)
                    if re_find:
                        result['suppress_interval'] = re_find
                        if re_find[0] != suppress_interval:
                            need_cfg = True
                    else:
                        need_cfg = True
            elif '<data/>' in recv_xml:
                pass
            else:
                re_find = re.findall('.*<suppressInterval>(.*)</suppressInterval>.*', recv_xml)
                if re_find:
                    result['suppress_interval'] = re_find
                    if re_find[0] == suppress_interval:
                        need_cfg = True
    if hold_interval:
        if not suppress_interval or not clear_interval:
            module.fail_json(msg='Error: Please input suppress_interval hold_interval clear_interval at the same time.')
        if int(hold_interval) > 65535 or int(hold_interval) < 1:
            module.fail_json(msg='Error: The hold_interval %s is out of [1 - 65535].' % hold_interval)
        else:
            conf_str = CE_GET_BGP_ENABLE_HEADER + '<holdInterval></holdInterval>' + CE_GET_BGP_ENABLE_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if state == 'present':
                if '<data/>' in recv_xml:
                    need_cfg = True
                else:
                    re_find = re.findall('.*<holdInterval>(.*)</holdInterval>.*', recv_xml)
                    if re_find:
                        result['hold_interval'] = re_find
                        if re_find[0] != hold_interval:
                            need_cfg = True
                    else:
                        need_cfg = True
            elif '<data/>' in recv_xml:
                pass
            else:
                re_find = re.findall('.*<holdInterval>(.*)</holdInterval>.*', recv_xml)
                if re_find:
                    result['hold_interval'] = re_find
                    if re_find[0] == hold_interval:
                        need_cfg = True
    if clear_interval:
        if not suppress_interval or not hold_interval:
            module.fail_json(msg='Error: Please input suppress_interval hold_interval clear_interval at the same time.')
        if int(clear_interval) > 65535 or int(clear_interval) < 1:
            module.fail_json(msg='Error: The clear_interval %s is out of [1 - 65535].' % clear_interval)
        else:
            conf_str = CE_GET_BGP_ENABLE_HEADER + '<clearInterval></clearInterval>' + CE_GET_BGP_ENABLE_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if state == 'present':
                if '<data/>' in recv_xml:
                    need_cfg = True
                else:
                    re_find = re.findall('.*<clearInterval>(.*)</clearInterval>.*', recv_xml)
                    if re_find:
                        result['clear_interval'] = re_find
                        if re_find[0] != clear_interval:
                            need_cfg = True
                    else:
                        need_cfg = True
            elif '<data/>' in recv_xml:
                pass
            else:
                re_find = re.findall('.*<clearInterval>(.*)</clearInterval>.*', recv_xml)
                if re_find:
                    result['clear_interval'] = re_find
                    if re_find[0] == clear_interval:
                        need_cfg = True
    result['need_cfg'] = need_cfg
    return result