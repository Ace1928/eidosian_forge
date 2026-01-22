import time
import logging
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.tests.integrated import tester
def test_flow_removed_idle_timeout(self, dp):
    reason = dp.ofproto.OFPRR_IDLE_TIMEOUT
    idle_timeout = 2
    self._add_flow_flow_removed(dp, reason, idle_timeout=idle_timeout)