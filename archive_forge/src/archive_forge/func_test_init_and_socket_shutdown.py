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
def test_init_and_socket_shutdown(self):
    """
        Test initialization of a socket manager with a failed shutdown.
        """
    self.assertFalse(self.fake_socket.connected)
    nop_called = False

    def nop(*args):
        nonlocal nop_called
        nop_called = True
    server_death_called = False

    def server_death(*args):
        nonlocal server_death_called
        server_death_called = True
    socket_manager = SocketManager('https://127.0.0.1', self.fake_socket.port, nop, nop, nop, TASK_GROUP_ID_1, 0.4, server_death)
    self.assertTrue(self.fake_socket.connected)
    self.assertFalse(nop_called)
    self.assertFalse(server_death_called)
    self.assertFalse(self.fake_socket.disconnected)
    self.assertFalse(socket_manager.is_shutdown)
    self.assertTrue(socket_manager.alive)
    self.fake_socket.close()
    self.assertEqualBy(lambda: socket_manager.alive, False, 8)
    self.assertEqualBy(lambda: server_death_called, True, 20)
    self.assertFalse(nop_called)
    socket_manager.shutdown()