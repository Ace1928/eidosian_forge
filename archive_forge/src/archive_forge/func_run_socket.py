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