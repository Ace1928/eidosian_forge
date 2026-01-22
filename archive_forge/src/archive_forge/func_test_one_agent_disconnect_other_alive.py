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
def test_one_agent_disconnect_other_alive(self):
    message_packet = None

    def on_msg(*args):
        nonlocal message_packet
        message_packet = args[0]
    self.agent1.register_to_socket(self.fake_socket, on_msg)
    self.agent2.register_to_socket(self.fake_socket, on_msg)
    self.assertIsNone(message_packet)
    self.agent1.send_alive()
    self.agent2.send_alive()
    self.assertIsNone(message_packet)
    self.agent2.send_disconnect()
    self.assertEqualBy(lambda: self.dead_worker_id, TEST_WORKER_ID_2, 8)
    self.assertEqual(self.dead_assignment_id, TEST_ASSIGNMENT_ID_2)
    test_message_text_1 = 'test_message_text_1'
    msg_id = self.agent1.send_message(test_message_text_1)
    self.assertEqualBy(lambda: self.message_packet is None, False, 8)
    self.assertEqual(self.message_packet.id, msg_id)
    self.assertEqual(self.message_packet.data['text'], test_message_text_1)
    manager_message_id = 'message_id_from_manager'
    test_message_text_2 = 'test_message_text_2'
    message_send_packet = Packet(manager_message_id, data_model.WORLD_MESSAGE, self.socket_manager.get_my_sender_id(), TEST_WORKER_ID_1, TEST_ASSIGNMENT_ID_1, test_message_text_2, 't2')
    self.socket_manager.queue_packet(message_send_packet)
    self.assertEqualBy(lambda: message_packet is None, False, 8)
    self.assertEqual(message_packet.id, manager_message_id)
    self.assertEqual(message_packet.data, test_message_text_2)
    self.assertIn(manager_message_id, self.socket_manager.packet_map)
    self.agent1.send_disconnect()
    self.assertEqualBy(lambda: self.dead_worker_id, TEST_WORKER_ID_1, 8)
    self.assertEqual(self.dead_assignment_id, TEST_ASSIGNMENT_ID_1)