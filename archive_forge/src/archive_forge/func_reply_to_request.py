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
def reply_to_request(self, req, rep):
    """
        Send a reply for a synchronous request sent by send_request.
        The first argument should be an instance of EventRequestBase.
        The second argument should be an instance of EventReplyBase.
        """
    assert isinstance(req, EventRequestBase)
    assert isinstance(rep, EventReplyBase)
    rep.dst = req.src
    if req.sync:
        req.reply_q.put(rep)
    else:
        self.send_event(rep.dst, rep)