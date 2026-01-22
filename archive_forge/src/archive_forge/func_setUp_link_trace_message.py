import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import cfm
def setUp_link_trace_message(self):
    self.link_trace_message_md_lv = 1
    self.link_trace_message_version = 1
    self.link_trace_message_use_fdb_only = 1
    self.link_trace_message_transaction_id = 12345
    self.link_trace_message_ttl = 123
    self.link_trace_message_ltm_orig_addr = '11:22:33:44:55:66'
    self.link_trace_message_ltm_targ_addr = '77:88:99:aa:cc:dd'
    self.link_trace_message_tlvs = [cfm.sender_id_tlv(), cfm.port_status_tlv(), cfm.data_tlv(), cfm.interface_status_tlv(), cfm.reply_ingress_tlv(), cfm.reply_egress_tlv(), cfm.ltm_egress_identifier_tlv(), cfm.ltr_egress_identifier_tlv(), cfm.organization_specific_tlv()]
    self.message = cfm.link_trace_message(self.link_trace_message_md_lv, self.link_trace_message_version, self.link_trace_message_use_fdb_only, self.link_trace_message_transaction_id, self.link_trace_message_ttl, self.link_trace_message_ltm_orig_addr, self.link_trace_message_ltm_targ_addr, self.link_trace_message_tlvs)
    self.ins = cfm.cfm(self.message)
    data = bytearray()
    prev = None
    self.buf = self.ins.serialize(data, prev)