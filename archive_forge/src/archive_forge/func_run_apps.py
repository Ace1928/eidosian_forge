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
@staticmethod
def run_apps(app_lists):
    """Run a set of OSKen applications

        A convenient method to load and instantiate apps.
        This blocks until all relevant apps stop.
        """
    app_mgr = AppManager.get_instance()
    app_mgr.load_apps(app_lists)
    contexts = app_mgr.create_contexts()
    services = app_mgr.instantiate_apps(**contexts)
    try:
        hub.joinall(services)
    finally:
        app_mgr.close()
        for t in services:
            t.kill()
        hub.joinall(services)
        gc.collect()