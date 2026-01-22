import unittest
import time
import uuid
from unittest import mock
from parlai.mturk.core.dev.socket_manager import Packet, SocketManager
from parlai.mturk.core.dev.agents import AssignState
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
import threading
from websocket_server import WebsocketServer
import json
def make_packet_handler(self, on_msg):
    """
        A packet handler.
        """

    def handler_mock(pkt):
        if pkt['type'] == data_model.WORLD_MESSAGE:
            packet = Packet.from_dict(pkt)
            on_msg(packet)
        elif pkt['type'] == data_model.MESSAGE_BATCH:
            packet = Packet.from_dict(pkt)
            on_msg(packet)
        elif pkt['type'] == data_model.AGENT_ALIVE:
            raise Exception('Invalid alive packet {}'.format(pkt))
        else:
            raise Exception('Invalid Packet type {} received in {}'.format(pkt['type'], pkt))
    return handler_mock