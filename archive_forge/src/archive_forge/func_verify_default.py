import time
import logging
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.tests.integrated import tester
def verify_default(self, dp, msg):
    type_ = self._verify
    if msg.msg_type == dp.ofproto.OFPT_STATS_REPLY:
        return self.verify_stats(dp, msg.body, type_)
    elif msg.msg_type == type_:
        return True
    else:
        return 'Reply msg_type %s expected %s' % (msg.msg_type, type_)