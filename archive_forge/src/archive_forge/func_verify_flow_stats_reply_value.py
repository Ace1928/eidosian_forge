import time
import logging
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.tests.integrated import tester
def verify_flow_stats_reply_value(self, dp, msg):
    c = 0
    for f in msg.body:
        f_value = (f.table_id, f.cookie, f.idle_timeout, f.hard_timeout, f.priority)
        if f_value != self._verify[c]:
            return 'param is mismatched. verify=%s, reply=%s' % (self._verify[c], f_value)
        c += 1
    return len(msg.body) == self.n_tables