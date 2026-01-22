import unittest
import time
import uuid
from unittest import mock
from parlai.mturk.core.socket_manager import Packet, SocketManager
from parlai.mturk.core.shared_utils import AssignState
import parlai.mturk.core.data_model as data_model
import parlai.mturk.core.shared_utils as shared_utils
import threading
from websocket_server import WebsocketServer
import json
def send_heartbeat(self):
    """
        Sends a heartbeat to the world.
        """
    hb = {'id': str(uuid.uuid4()), 'receiver_id': '[World_' + self.task_group_id + ']', 'assignment_id': self.assignment_id, 'sender_id': self.worker_id, 'conversation_id': self.conversation_id, 'type': Packet.TYPE_HEARTBEAT, 'data': None}
    self.ws.send(json.dumps({'type': data_model.SOCKET_ROUTE_PACKET_STRING, 'content': hb}))