import unittest
import os
import time
import json
import threading
import pickle
from unittest import mock
from parlai.mturk.core.dev.worker_manager import WorkerManager
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
from parlai.mturk.core.dev.worlds import MTurkOnboardWorld
from parlai.mturk.core.dev.mturk_manager import MTurkManager
from parlai.mturk.core.dev.socket_manager import SocketManager, Packet
from parlai.core.params import ParlaiParser
from websocket_server import WebsocketServer
import parlai.mturk.core.dev.mturk_manager as MTurkManagerFile
import parlai.mturk.core.dev.data_model as data_model
def launch_socket(self):

    def on_message(client, server, message):
        if self.closed:
            raise Exception('Socket is already closed...')
        if message == '':
            return
        packet_dict = json.loads(message)
        if packet_dict['content']['id'] == 'WORLD_ALIVE':
            self.ws.send_message(client, json.dumps({'type': 'conn_success'}))
            self.connected = True
        elif packet_dict['type'] == data_model.WORLD_PING:
            pong = packet_dict['content'].copy()
            pong['type'] = 'pong'
            self.ws.send_message(client, json.dumps({'type': data_model.SERVER_PONG, 'content': pong}))
        if 'receiver_id' in packet_dict['content']:
            receiver_id = packet_dict['content']['receiver_id']
            use_func = self.handlers.get(receiver_id, self.do_nothing)
            use_func(packet_dict['content'])

    def on_connect(client, server):
        pass

    def on_disconnect(client, server):
        self.disconnected = True

    def run_socket(*args):
        port = 3030
        while self.port is None:
            try:
                self.ws = WebsocketServer(port, host='127.0.0.1')
                self.port = port
            except OSError:
                port += 1
        self.ws.set_fn_client_left(on_disconnect)
        self.ws.set_fn_new_client(on_connect)
        self.ws.set_fn_message_received(on_message)
        self.ws.run_forever()
    self.listen_thread = threading.Thread(target=run_socket, name='Fake-Socket-Thread')
    self.listen_thread.daemon = True
    self.listen_thread.start()