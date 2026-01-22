import unittest
import os
from unittest import mock
from parlai.mturk.core.dev.worker_manager import WorkerManager, WorkerState
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
from parlai.mturk.core.dev.mturk_manager import MTurkManager
from parlai.core.params import ParlaiParser
import parlai.mturk.core.dev.worker_manager as WorkerManagerFile
import parlai.mturk.core.dev.data_model as data_model
def test_time_blocks(self):
    """
        Check to see if time blocking and clearing works.
        """
    self.mturk_manager.soft_block_worker = mock.MagicMock()
    self.mturk_manager.un_soft_block_worker = mock.MagicMock()
    self.worker_manager.un_time_block_workers()
    self.assertEqual(len(self.mturk_manager.un_soft_block_worker.mock_calls), 0)
    self.assertEqual(len(self.worker_manager.time_blocked_workers), 0)
    self.worker_manager.time_block_worker(TEST_WORKER_ID_1)
    self.mturk_manager.soft_block_worker.assert_called_with(TEST_WORKER_ID_1, 'max_time_qual')
    self.assertEqual(len(self.worker_manager.time_blocked_workers), 1)
    self.worker_manager.time_block_worker(TEST_WORKER_ID_2)
    self.mturk_manager.soft_block_worker.assert_called_with(TEST_WORKER_ID_2, 'max_time_qual')
    self.assertEqual(len(self.worker_manager.time_blocked_workers), 2)
    self.assertEqual(len(self.mturk_manager.soft_block_worker.mock_calls), 2)
    self.worker_manager.un_time_block_workers([TEST_WORKER_ID_3])
    self.mturk_manager.un_soft_block_worker.assert_called_with(TEST_WORKER_ID_3, 'max_time_qual')
    self.assertEqual(len(self.worker_manager.time_blocked_workers), 2)
    self.worker_manager.un_time_block_workers()
    self.assertEqual(len(self.worker_manager.time_blocked_workers), 0)
    self.mturk_manager.un_soft_block_worker.assert_any_call(TEST_WORKER_ID_1, 'max_time_qual')
    self.mturk_manager.un_soft_block_worker.assert_any_call(TEST_WORKER_ID_2, 'max_time_qual')