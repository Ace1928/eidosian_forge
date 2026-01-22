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
def test_pool_add_get_remove_and_expire(self):
    """
        Ensure the pool properly adds and releases workers.
        """
    all_are_eligible = {'multiple': True, 'func': lambda workers: workers}
    manager = self.mturk_manager
    pool = manager._get_unique_pool(all_are_eligible)
    self.assertEqual(pool, [])
    manager._add_agent_to_pool(self.agent_1)
    manager._add_agent_to_pool(self.agent_2)
    manager._add_agent_to_pool(self.agent_3)
    self.assertListEqual(manager._get_unique_pool(all_are_eligible), [self.agent_1, self.agent_2, self.agent_3])
    manager._add_agent_to_pool(self.agent_1)
    self.assertListEqual(manager._get_unique_pool(all_are_eligible), [self.agent_1, self.agent_2, self.agent_3])
    manager._remove_from_agent_pool(self.agent_2)
    self.assertListEqual(manager._get_unique_pool(all_are_eligible), [self.agent_1, self.agent_3])
    with self.assertRaises(AssertionError):
        manager._remove_from_agent_pool(self.agent_2)
    second_worker_only = {'multiple': True, 'func': lambda workers: [workers[1]]}
    self.assertListEqual(manager._get_unique_pool(second_worker_only), [self.agent_3])
    only_agent_1 = {'multiple': False, 'func': lambda worker: worker is self.agent_1}
    self.assertListEqual(manager._get_unique_pool(only_agent_1), [self.agent_1])
    manager.force_expire_hit = mock.MagicMock()
    manager._expire_agent_pool()
    manager.force_expire_hit.assert_any_call(self.agent_1.worker_id, self.agent_1.assignment_id)
    manager.force_expire_hit.assert_any_call(self.agent_3.worker_id, self.agent_3.assignment_id)
    pool = manager._get_unique_pool(all_are_eligible)
    self.assertEqual(pool, [])
    self.agent_2.worker_id = self.agent_1.worker_id
    manager._add_agent_to_pool(self.agent_1)
    manager._add_agent_to_pool(self.agent_2)
    self.assertListEqual(manager.agent_pool, [self.agent_1, self.agent_2])
    manager.is_sandbox = False
    self.assertListEqual(manager._get_unique_pool(all_are_eligible), [self.agent_1])