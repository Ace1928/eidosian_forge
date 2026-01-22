import unittest
import time
import uuid
import os
from unittest import mock
from parlai.mturk.core.socket_manager import Packet, SocketManager
from parlai.mturk.core.shared_utils import AssignState
from parlai.mturk.core.mturk_manager import MTurkManager
from parlai.core.params import ParlaiParser
import parlai.mturk.core.mturk_manager as MTurkManagerFile
import parlai.mturk.core.data_model as data_model
import parlai.mturk.core.shared_utils as shared_utils
import threading
from websocket_server import WebsocketServer
import json
def onboard_agent(self, worker):
    self.onboarding_agents[worker.worker_id] = False
    while worker.worker_id in self.onboarding_agents and self.onboarding_agents[worker.worker_id] is False:
        time.sleep(0.05)
    return