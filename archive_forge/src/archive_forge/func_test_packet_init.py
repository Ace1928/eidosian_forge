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
def test_packet_init(self):
    """
        Test proper initialization of packet fields.
        """
    self.assertEqual(self.packet_1.id, self.ID)
    self.assertEqual(self.packet_1.type, data_model.MESSAGE_BATCH)
    self.assertEqual(self.packet_1.sender_id, self.SENDER_ID)
    self.assertEqual(self.packet_1.receiver_id, self.RECEIVER_ID)
    self.assertEqual(self.packet_1.assignment_id, self.ASSIGNMENT_ID)
    self.assertEqual(self.packet_1.data, self.DATA)
    self.assertEqual(self.packet_1.conversation_id, self.CONVERSATION_ID)
    self.assertEqual(self.packet_1.ack_func, self.ACK_FUNCTION)
    self.assertEqual(self.packet_1.status, Packet.STATUS_INIT)
    self.assertEqual(self.packet_2.id, self.ID)
    self.assertEqual(self.packet_2.type, data_model.SNS_MESSAGE)
    self.assertEqual(self.packet_2.sender_id, self.SENDER_ID)
    self.assertEqual(self.packet_2.receiver_id, self.RECEIVER_ID)
    self.assertEqual(self.packet_2.assignment_id, self.ASSIGNMENT_ID)
    self.assertEqual(self.packet_2.data, self.DATA)
    self.assertIsNone(self.packet_2.conversation_id)
    self.assertIsNone(self.packet_2.ack_func)
    self.assertEqual(self.packet_2.status, Packet.STATUS_INIT)
    self.assertEqual(self.packet_3.id, self.ID)
    self.assertEqual(self.packet_3.type, data_model.AGENT_ALIVE)
    self.assertEqual(self.packet_3.sender_id, self.SENDER_ID)
    self.assertEqual(self.packet_3.receiver_id, self.RECEIVER_ID)
    self.assertEqual(self.packet_3.assignment_id, self.ASSIGNMENT_ID)
    self.assertEqual(self.packet_3.data, self.DATA)
    self.assertIsNone(self.packet_3.conversation_id)
    self.assertIsNone(self.packet_3.ack_func)
    self.assertEqual(self.packet_3.status, Packet.STATUS_INIT)