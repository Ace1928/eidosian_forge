import datetime
import logging
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import handler
from os_ken.controller import ofp_event
from os_ken.controller.handler import set_ev_cls
from os_ken.exception import OSKenException
from os_ken.exception import OFPUnknownVersion
from os_ken.lib import hub
from os_ken.lib import mac
from os_ken.lib.dpid import dpid_to_str
from os_ken.lib.packet import bpdu
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import llc
from os_ken.lib.packet import packet
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_3
@staticmethod
def compare_bpdu_info(my_priority, my_times, rcv_priority, rcv_times):
    """ Check received BPDU is superior to currently held BPDU
             by the following comparison.
             - root bridge ID value
             - root path cost
             - designated bridge ID value
             - designated port ID value
             - times """
    if my_priority is None:
        result = SUPERIOR
    else:
        result = Stp._cmp_value(rcv_priority.root_id.value, my_priority.root_id.value)
        if not result:
            result = Stp.compare_root_path(rcv_priority.root_path_cost, my_priority.root_path_cost, rcv_priority.designated_bridge_id.value, my_priority.designated_bridge_id.value, rcv_priority.designated_port_id.value, my_priority.designated_port_id.value)
            if not result:
                result1 = Stp._cmp_value(rcv_priority.designated_bridge_id.value, mac.haddr_to_int(my_priority.designated_bridge_id.mac_addr))
                result2 = Stp._cmp_value(rcv_priority.designated_port_id.value, my_priority.designated_port_id.port_no)
                if not result1 and (not result2):
                    result = SUPERIOR
                else:
                    result = Stp._cmp_obj(rcv_times, my_times)
    return result