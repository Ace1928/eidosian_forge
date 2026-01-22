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
def build_and_send_packet(self, packet_type, data):
    msg_id = str(uuid.uuid4())
    msg = {'id': msg_id, 'type': packet_type, 'sender_id': self.worker_id, 'assignment_id': self.assignment_id, 'conversation_id': self.conversation_id, 'receiver_id': '[World_' + self.task_group_id + ']', 'data': data}
    if packet_type == data_model.MESSAGE_BATCH:
        msg['data'] = {'messages': [{'id': msg_id, 'type': packet_type, 'sender_id': self.worker_id, 'assignment_id': self.assignment_id, 'conversation_id': self.conversation_id, 'receiver_id': '[World_' + self.task_group_id + ']', 'data': data}]}
    self.ws.send(json.dumps({'type': packet_type, 'content': msg}))
    return msg['id']