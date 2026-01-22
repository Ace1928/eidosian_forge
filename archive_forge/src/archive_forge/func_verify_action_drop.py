import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def verify_action_drop(self, dp, stats):
    for s in stats:
        for i in s.instructions:
            if len(i.actions):
                return 'has actions. %s' % i.actions
    return True