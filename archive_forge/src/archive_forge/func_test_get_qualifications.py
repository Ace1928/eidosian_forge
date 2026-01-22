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
def test_get_qualifications(self):
    manager = self.mturk_manager
    mturk_utils = MTurkManagerFile.mturk_utils
    mturk_utils.find_or_create_qualification = mock.MagicMock()
    fake_qual = {'QualificationTypeId': 'fake_qual_id', 'Comparator': 'DoesNotExist', 'ActionsGuarded': 'DiscoverPreviewAndAccept'}
    qualifications = manager.get_qualification_list([fake_qual])
    self.assertListEqual(qualifications, [fake_qual])
    self.assertListEqual(manager.qualifications, [fake_qual])
    mturk_utils.find_or_create_qualification.assert_not_called()
    disconnect_qual_name = 'disconnect_qual_name'
    disconnect_qual_id = 'disconnect_qual_id'
    block_qual_name = 'block_qual_name'
    block_qual_id = 'block_qual_id'
    max_time_qual_name = 'max_time_qual_name'
    max_time_qual_id = 'max_time_qual_id'
    unique_qual_name = 'unique_qual_name'
    unique_qual_id = 'unique_qual_id'

    def return_qualifications(qual_name, _text, _sb):
        if qual_name == disconnect_qual_name:
            return disconnect_qual_id
        if qual_name == block_qual_name:
            return block_qual_id
        if qual_name == max_time_qual_name:
            return max_time_qual_id
        if qual_name == unique_qual_name:
            return unique_qual_id
    mturk_utils.find_or_create_qualification = return_qualifications
    manager.opt['disconnect_qualification'] = disconnect_qual_name
    manager.opt['block_qualification'] = block_qual_name
    manager.opt['max_time_qual'] = max_time_qual_name
    manager.opt['unique_qual_name'] = unique_qual_name
    manager.is_unique = True
    manager.has_time_limit = True
    manager.qualifications = None
    qualifications = manager.get_qualification_list()
    for qual in qualifications:
        self.assertEqual(qual['ActionsGuarded'], 'DiscoverPreviewAndAccept')
        self.assertEqual(qual['Comparator'], 'DoesNotExist')
    for qual_id in [disconnect_qual_id, block_qual_id, max_time_qual_id, unique_qual_id]:
        has_qual = False
        for qual in qualifications:
            if qual['QualificationTypeId'] == qual_id:
                has_qual = True
                break
        self.assertTrue(has_qual)
    self.assertListEqual(qualifications, manager.qualifications)