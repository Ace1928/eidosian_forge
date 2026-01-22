import logging
import math
import os
import pickle
import threading
import time
import uuid
import errno
import requests
from parlai.mturk.core.shared_utils import AssignState
from parlai.mturk.core.socket_manager import Packet, SocketManager, StaticSocketManager
from parlai.mturk.core.worker_manager import WorkerManager
from parlai.mturk.core.mturk_data_handler import MTurkDataHandler
import parlai.mturk.core.data_model as data_model
import parlai.mturk.core.mturk_utils as mturk_utils
import parlai.mturk.core.server_utils as server_utils
import parlai.mturk.core.shared_utils as shared_utils
def populate_legacy_task_files(self, task_directory_path):
    if not self.task_files_to_copy:
        self.task_files_to_copy = []
    if not task_directory_path:
        task_directory_path = os.path.join(self.opt['parlai_home'], 'parlai', 'mturk', 'tasks', self.opt['task'])
    self.task_files_to_copy.append(os.path.join(task_directory_path, 'html', 'cover_page.html'))
    try:
        for file_name in os.listdir(os.path.join(task_directory_path, 'html')):
            self.task_files_to_copy.append(os.path.join(task_directory_path, 'html', file_name))
    except FileNotFoundError:
        pass
    for mturk_agent_id in self.mturk_agent_ids + ['onboarding']:
        self.task_files_to_copy.append(os.path.join(task_directory_path, 'html', '{}_index.html'.format(mturk_agent_id)))