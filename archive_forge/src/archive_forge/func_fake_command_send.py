import unittest
import os
from unittest import mock
from parlai.mturk.core.dev.worker_manager import WorkerManager, WorkerState
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
from parlai.mturk.core.dev.mturk_manager import MTurkManager
from parlai.core.params import ParlaiParser
import parlai.mturk.core.dev.worker_manager as WorkerManagerFile
import parlai.mturk.core.dev.data_model as data_model
def fake_command_send(worker_id, assignment_id, data, ack_func):
    pkt = mock.MagicMock()
    pkt.sender_id = worker_id
    pkt.assignment_id = assignment_id
    self.assertEqual(data['text'], data_model.COMMAND_CHANGE_CONVERSATION)
    ack_func(pkt)