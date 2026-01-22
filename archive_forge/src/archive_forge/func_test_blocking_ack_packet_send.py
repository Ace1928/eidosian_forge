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
def test_blocking_ack_packet_send(self):
    """
        Checks to see if ack'ed blocking packets are working properly.
        """
    self.socket_manager._safe_send = mock.MagicMock()
    self.socket_manager._safe_put = mock.MagicMock()
    self.sent = False
    send_time = time.time()
    self.assertEqual(self.MESSAGE_SEND_PACKET_1.status, Packet.STATUS_INIT)
    self._send_packet_in_background(self.MESSAGE_SEND_PACKET_1, send_time)
    self.assertEqual(self.MESSAGE_SEND_PACKET_1.status, Packet.STATUS_SENT)
    self.socket_manager._safe_send.assert_called_once()
    connection_id = self.MESSAGE_SEND_PACKET_1.get_receiver_connection_id()
    self.socket_manager._safe_put.assert_called_once_with(connection_id, (send_time, self.MESSAGE_SEND_PACKET_1))
    self.assertTrue(self.sent)
    self.socket_manager._safe_send.reset_mock()
    self.socket_manager._safe_put.reset_mock()
    self.MESSAGE_SEND_PACKET_1.status = Packet.STATUS_ACK
    self._send_packet_in_background(self.MESSAGE_SEND_PACKET_1, send_time)
    self.socket_manager._safe_send.assert_not_called()
    self.socket_manager._safe_put.assert_not_called()