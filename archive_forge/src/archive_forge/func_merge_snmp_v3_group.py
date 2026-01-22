from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def merge_snmp_v3_group(self, **kwargs):
    """ Merge snmp v3 group operation """
    module = kwargs['module']
    group_name = module.params['group_name']
    security_level = module.params['security_level']
    acl_number = module.params['acl_number']
    read_view = module.params['read_view']
    write_view = module.params['write_view']
    notify_view = module.params['notify_view']
    conf_str = CE_MERGE_SNMP_V3_GROUP_HEADER % (group_name, security_level)
    if acl_number:
        conf_str += '<aclNumber>%s</aclNumber>' % acl_number
    if read_view:
        conf_str += '<readViewName>%s</readViewName>' % read_view
    if write_view:
        conf_str += '<writeViewName>%s</writeViewName>' % write_view
    if notify_view:
        conf_str += '<notifyViewName>%s</notifyViewName>' % notify_view
    conf_str += CE_MERGE_SNMP_V3_GROUP_TAIL
    recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
    if '<ok/>' not in recv_xml:
        module.fail_json(msg='Error: Merge snmp v3 group failed.')
    if security_level == 'noAuthNoPriv':
        security_level_cli = 'noauthentication'
    elif security_level == 'authentication':
        security_level_cli = 'authentication'
    elif security_level == 'privacy':
        security_level_cli = 'privacy'
    cmd = 'snmp-agent group v3 %s %s' % (group_name, security_level_cli)
    if read_view:
        cmd += ' read-view %s' % read_view
    if write_view:
        cmd += ' write-view %s' % write_view
    if notify_view:
        cmd += ' notify-view %s' % notify_view
    if acl_number:
        cmd += ' acl %s' % acl_number
    return cmd