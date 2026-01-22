import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import cfm
def setUp_link_trace_reply(self):
    self.link_trace_reply_md_lv = 1
    self.link_trace_reply_version = 1
    self.link_trace_reply_use_fdb_only = 1
    self.link_trace_reply_fwd_yes = 0
    self.link_trace_reply_terminal_mep = 1
    self.link_trace_reply_transaction_id = 5432
    self.link_trace_reply_ttl = 123
    self.link_trace_reply_relay_action = 3
    self.link_trace_reply_tlvs = [cfm.sender_id_tlv(), cfm.port_status_tlv(), cfm.data_tlv(), cfm.interface_status_tlv(), cfm.reply_ingress_tlv(), cfm.reply_egress_tlv(), cfm.ltm_egress_identifier_tlv(), cfm.ltr_egress_identifier_tlv(), cfm.organization_specific_tlv()]
    self.message = cfm.link_trace_reply(self.link_trace_reply_md_lv, self.link_trace_reply_version, self.link_trace_reply_use_fdb_only, self.link_trace_reply_fwd_yes, self.link_trace_reply_terminal_mep, self.link_trace_reply_transaction_id, self.link_trace_reply_ttl, self.link_trace_reply_relay_action, self.link_trace_reply_tlvs)
    self.ins = cfm.cfm(self.message)
    data = bytearray()
    prev = None
    self.buf = self.ins.serialize(data, prev)