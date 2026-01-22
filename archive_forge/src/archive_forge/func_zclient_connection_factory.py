import contextlib
import logging
import os
import socket
import struct
from os_ken import cfg
from os_ken.base import app_manager
from os_ken.base.app_manager import OSKenApp
from os_ken.controller.handler import set_ev_cls
from os_ken.lib import hub
from os_ken.lib import ip
from os_ken.lib.packet import zebra
from os_ken.services.protocols.zebra import db
from os_ken.services.protocols.zebra import event
from os_ken.services.protocols.zebra.server import event as zserver_event
def zclient_connection_factory(sock, addr):
    LOG.debug('Connected from client: %s: %s', addr, sock)
    zserv = app_manager.lookup_service_brick(ZServer.__name__)
    with contextlib.closing(ZClient(zserv, sock, addr)) as zclient:
        try:
            zclient.start()
        except Exception as e:
            LOG.error('Error in client%s: %s', addr, e)
            raise e