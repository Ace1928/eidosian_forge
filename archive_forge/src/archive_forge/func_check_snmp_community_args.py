from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def check_snmp_community_args(self, **kwargs):
    """ Check snmp community args """
    module = kwargs['module']
    result = dict()
    need_cfg = False
    result['community_info'] = []
    state = module.params['state']
    community_name = module.params['community_name']
    access_right = module.params['access_right']
    acl_number = module.params['acl_number']
    community_mib_view = module.params['community_mib_view']
    if community_name and access_right:
        if len(community_name) > 32 or len(community_name) == 0:
            module.fail_json(msg='Error: The len of community_name %s is out of [1 - 32].' % community_name)
        if acl_number:
            if acl_number.isdigit():
                if int(acl_number) > 2999 or int(acl_number) < 2000:
                    module.fail_json(msg='Error: The value of acl_number %s is out of [2000 - 2999].' % acl_number)
            elif not acl_number[0].isalpha() or len(acl_number) > 32 or len(acl_number) < 1:
                module.fail_json(msg='Error: The len of acl_number %s is out of [1 - 32] or is invalid.' % acl_number)
        if community_mib_view:
            if len(community_mib_view) > 32 or len(community_mib_view) == 0:
                module.fail_json(msg='Error: The len of community_mib_view %s is out of [1 - 32].' % community_mib_view)
        conf_str = CE_GET_SNMP_COMMUNITY_HEADER
        if acl_number:
            conf_str += '<aclNumber></aclNumber>'
        if community_mib_view:
            conf_str += '<mibViewName></mibViewName>'
        conf_str += CE_GET_SNMP_COMMUNITY_TAIL
        recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
        if '<data/>' in recv_xml:
            if state == 'present':
                need_cfg = True
        else:
            xml_str = recv_xml.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
            root = ElementTree.fromstring(xml_str)
            community_info = root.findall('snmp/communitys/community')
            if community_info:
                for tmp in community_info:
                    tmp_dict = dict()
                    for site in tmp:
                        if site.tag in ['communityName', 'accessRight', 'aclNumber', 'mibViewName']:
                            tmp_dict[site.tag] = site.text
                    result['community_info'].append(tmp_dict)
            if result['community_info']:
                community_name_list = list()
                for tmp in result['community_info']:
                    if 'communityName' in tmp.keys():
                        community_name_list.append(tmp['communityName'])
                if community_name not in community_name_list:
                    need_cfg = True
                else:
                    need_cfg_bool = True
                    for tmp in result['community_info']:
                        if tmp['communityName'] == community_name:
                            cfg_bool_list = list()
                            if access_right:
                                if 'accessRight' in tmp.keys():
                                    need_cfg_access = False
                                    if tmp['accessRight'] != access_right:
                                        need_cfg_access = True
                                else:
                                    need_cfg_access = True
                                cfg_bool_list.append(need_cfg_access)
                            if acl_number:
                                if 'aclNumber' in tmp.keys():
                                    need_cfg_acl = False
                                    if tmp['aclNumber'] != acl_number:
                                        need_cfg_acl = True
                                else:
                                    need_cfg_acl = True
                                cfg_bool_list.append(need_cfg_acl)
                            if community_mib_view:
                                if 'mibViewName' in tmp.keys():
                                    need_cfg_mib = False
                                    if tmp['mibViewName'] != community_mib_view:
                                        need_cfg_mib = True
                                else:
                                    need_cfg_mib = True
                                cfg_bool_list.append(need_cfg_mib)
                            if True not in cfg_bool_list:
                                need_cfg_bool = False
                    if state == 'present':
                        if not need_cfg_bool:
                            need_cfg = False
                        else:
                            need_cfg = True
                    elif not need_cfg_bool:
                        need_cfg = True
                    else:
                        need_cfg = False
    result['need_cfg'] = need_cfg
    return result