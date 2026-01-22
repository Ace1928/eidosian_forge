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
def detect_address_family(host):
    if ip.valid_ipv4(host):
        return socket.AF_INET
    elif ip.valid_ipv6(host):
        return socket.AF_INET6
    elif os.path.isdir(os.path.dirname(host)):
        return socket.AF_UNIX
    else:
        return None