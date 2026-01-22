import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import cfm
def setUp_loopback_message(self):
    self.loopback_message_md_lv = 1
    self.loopback_message_version = 1
    self.loopback_message_transaction_id = 12345
    self.loopback_message_tlvs = [cfm.sender_id_tlv(), cfm.port_status_tlv(), cfm.data_tlv(), cfm.interface_status_tlv(), cfm.reply_ingress_tlv(), cfm.reply_egress_tlv(), cfm.ltm_egress_identifier_tlv(), cfm.ltr_egress_identifier_tlv(), cfm.organization_specific_tlv()]
    self.message = cfm.loopback_message(self.loopback_message_md_lv, self.loopback_message_version, self.loopback_message_transaction_id, self.loopback_message_tlvs)
    self.ins = cfm.cfm(self.message)
    data = bytearray()
    prev = None
    self.buf = self.ins.serialize(data, prev)