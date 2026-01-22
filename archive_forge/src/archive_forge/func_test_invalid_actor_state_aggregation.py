import copy
import logging
from struct import pack, unpack_from
import unittest
from os_ken.ofproto import ether
from os_ken.lib.packet.ethernet import ethernet
from os_ken.lib.packet.packet import Packet
from os_ken.lib import addrconv
from os_ken.lib.packet.slow import slow, lacp
from os_ken.lib.packet.slow import SLOW_PROTOCOL_MULTICAST
from os_ken.lib.packet.slow import SLOW_SUBTYPE_LACP
from os_ken.lib.packet.slow import SLOW_SUBTYPE_MARKER
def test_invalid_actor_state_aggregation(self):
    self.assertRaises(AssertionError, lacp, self.version, self.actor_system_priority, self.actor_system, self.actor_key, self.actor_port_priority, self.actor_port, self.actor_state_activity, self.actor_state_timeout, 2, self.actor_state_synchronization, self.actor_state_collecting, self.actor_state_distributing, self.actor_state_defaulted, self.actor_state_expired, self.partner_system_priority, self.partner_system, self.partner_key, self.partner_port_priority, self.partner_port, self.partner_state_activity, self.partner_state_timeout, self.partner_state_aggregation, self.partner_state_synchronization, self.partner_state_collecting, self.partner_state_distributing, self.partner_state_defaulted, self.partner_state_expired, self.collector_max_delay)