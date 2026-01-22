from __future__ import (absolute_import, division, print_function)
import copy
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import set_nc_config, get_nc_config
def config_interface_tlv_disable_config(self):
    if self.function_lldp_interface_flag == 'tlvdisableINTERFACE':
        if self.enable_flag == 1 and self.conf_tlv_disable_exsit:
            if self.type_tlv_disable == 'basic_tlv':
                if self.ifname:
                    if self.portdesctxenable:
                        xml_str = CE_NC_MERGE_INTERFACE_TLV_CONFIG_HEADER % self.ifname + CE_NC_MERGE_INTERFACE_TLV_CONFIG_DISABLE_PORTDESCTXENABLE % self.portdesctxenable + CE_NC_MERGE_INTERFACE_TLV_CONFIG_TAIL
                        ret_xml = set_nc_config(self.module, xml_str)
                        self.check_response(ret_xml, 'TLV_DISABLE_PORTDESCTXENABLE')
                        self.changed = True
                    if self.manaddrtxenable:
                        xml_str = CE_NC_MERGE_INTERFACE_TLV_CONFIG_HEADER % self.ifname + CE_NC_MERGE_INTERFACE_TLV_CONFIG_DISABLE_MANADDRTXENABLE % self.manaddrtxenable + CE_NC_MERGE_INTERFACE_TLV_CONFIG_TAIL
                        ret_xml = set_nc_config(self.module, xml_str)
                        self.check_response(ret_xml, 'TLV_DISABLE_MANADDRTXENABLE')
                        self.changed = True
                    if self.syscaptxenable:
                        xml_str = CE_NC_MERGE_INTERFACE_TLV_CONFIG_HEADER % self.ifname + CE_NC_MERGE_INTERFACE_TLV_CONFIG_DISABLE_SYSCAPTXENABLE % self.syscaptxenable + CE_NC_MERGE_INTERFACE_TLV_CONFIG_TAIL
                        ret_xml = set_nc_config(self.module, xml_str)
                        self.check_response(ret_xml, 'TLV_DISABLE_SYSCAPTXENABLE')
                        self.changed = True
                    if self.sysdesctxenable:
                        xml_str = CE_NC_MERGE_INTERFACE_TLV_CONFIG_HEADER % self.ifname + CE_NC_MERGE_INTERFACE_TLV_CONFIG_DISABLE_SYSDESCTXENABLE % self.sysdesctxenable + CE_NC_MERGE_INTERFACE_TLV_CONFIG_TAIL
                        ret_xml = set_nc_config(self.module, xml_str)
                        self.check_response(ret_xml, 'TLV_DISABLE_SYSDESCTXENABLE')
                        self.changed = True
                    if self.sysnametxenable:
                        xml_str = CE_NC_MERGE_INTERFACE_TLV_CONFIG_HEADER % self.ifname + CE_NC_MERGE_INTERFACE_TLV_CONFIG_DISABLE_SYSNAMETXENABLE % self.sysnametxenable + CE_NC_MERGE_INTERFACE_TLV_CONFIG_TAIL
                        ret_xml = set_nc_config(self.module, xml_str)
                        self.check_response(ret_xml, 'TLV_DISABLE_SYSNAMETXENABLE')
                        self.changed = True
            if self.type_tlv_disable == 'dot3_tlv':
                if self.ifname:
                    if self.linkaggretxenable:
                        xml_str = CE_NC_MERGE_INTERFACE_TLV_CONFIG_HEADER % self.ifname + CE_NC_MERGE_INTERFACE_TLV_CONFIG_DISABLE_LINKAGGRETXENABLE % self.linkaggretxenable + CE_NC_MERGE_INTERFACE_TLV_CONFIG_TAIL
                        ret_xml = set_nc_config(self.module, xml_str)
                        self.check_response(ret_xml, 'TLV_DISABLE_LINKAGGRETXENABLE')
                        self.changed = True
                    if self.macphytxenable:
                        xml_str = CE_NC_MERGE_INTERFACE_TLV_CONFIG_HEADER % self.ifname + CE_NC_MERGE_INTERFACE_TLV_CONFIG_DISABLE_MACPHYTXENABLE % self.macphytxenable + CE_NC_MERGE_INTERFACE_TLV_CONFIG_TAIL
                        ret_xml = set_nc_config(self.module, xml_str)
                        self.check_response(ret_xml, 'TLV_DISABLE_MACPHYTXENABLE')
                        self.changed = True
                    if self.maxframetxenable:
                        xml_str = CE_NC_MERGE_INTERFACE_TLV_CONFIG_HEADER % self.ifname + CE_NC_MERGE_INTERFACE_TLV_CONFIG_DISABLE_MAXFRAMETXENABLE % self.maxframetxenable + CE_NC_MERGE_INTERFACE_TLV_CONFIG_TAIL
                        ret_xml = set_nc_config(self.module, xml_str)
                        self.check_response(ret_xml, 'TLV_DISABLE_MAXFRAMETXENABLE')
                        self.changed = True
                    if self.eee:
                        xml_str = CE_NC_MERGE_INTERFACE_TLV_CONFIG_HEADER % self.ifname + CE_NC_MERGE_INTERFACE_TLV_CONFIG_DISABLE_EEE % self.eee + CE_NC_MERGE_INTERFACE_TLV_CONFIG_TAIL
                        ret_xml = set_nc_config(self.module, xml_str)
                        self.check_response(ret_xml, 'TLV_DISABLE_EEE')
                        self.changed = True