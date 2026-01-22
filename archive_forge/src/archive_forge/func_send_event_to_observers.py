import inspect
import itertools
import logging
import sys
import os
import gc
from os_ken import cfg
from os_ken import utils
from os_ken.controller.handler import register_instance, get_dependent_services
from os_ken.controller.controller import Datapath
from os_ken.controller import event
from os_ken.controller.event import EventRequestBase, EventReplyBase
from os_ken.lib import hub
from os_ken.ofproto import ofproto_protocol
def send_event_to_observers(self, ev, state=None):
    """
        Send the specified event to all observers of this OSKenApp.
        """
    for observer in self.get_observers(ev, state):
        self.send_event(observer, ev, state)