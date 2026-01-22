from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec, check_ip_addr
def merge_acl(self):
    """ Merge acl operation """
    conf_str = CE_MERGE_ACL_HEADER % self.acl_name
    if self.acl_type:
        conf_str += '<aclType>%s</aclType>' % self.acl_type
    if self.acl_num:
        conf_str += '<aclNumber>%s</aclNumber>' % self.acl_num
    if self.acl_step:
        conf_str += '<aclStep>%s</aclStep>' % self.acl_step
    if self.acl_description:
        conf_str += '<aclDescription>%s</aclDescription>' % self.acl_description
    conf_str += CE_MERGE_ACL_TAIL
    recv_xml = self.netconf_set_config(conf_str=conf_str)
    if '<ok/>' not in recv_xml:
        self.module.fail_json(msg='Error: Merge acl failed.')
    if self.acl_name.isdigit():
        cmd = 'acl number %s' % self.acl_name
    elif self.acl_type and (not self.acl_num):
        cmd = 'acl name %s %s' % (self.acl_name, self.acl_type.lower())
    elif self.acl_type and self.acl_num:
        cmd = 'acl name %s number %s' % (self.acl_name, self.acl_num)
    elif not self.acl_type and self.acl_num:
        cmd = 'acl name %s number %s' % (self.acl_name, self.acl_num)
    self.updates_cmd.append(cmd)
    if self.acl_description:
        cmd = 'description %s' % self.acl_description
        self.updates_cmd.append(cmd)
    if self.acl_step:
        cmd = 'step %s' % self.acl_step
        self.updates_cmd.append(cmd)
    self.changed = True