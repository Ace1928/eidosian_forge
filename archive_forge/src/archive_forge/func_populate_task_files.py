import logging
import math
import os
import pickle
import threading
import time
import uuid
import errno
import requests
from parlai.mturk.core.dev.agents import (
from parlai.mturk.core.dev.socket_manager import Packet, SocketManager
from parlai.mturk.core.dev.worker_manager import WorkerManager
from parlai.mturk.core.dev.mturk_data_handler import MTurkDataHandler
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.mturk_utils as mturk_utils
import parlai.mturk.core.dev.server_utils as server_utils
import parlai.mturk.core.dev.shared_utils as shared_utils
def populate_task_files(self, task_directory_path):
    if not self.task_files_to_copy:
        self.task_files_to_copy = {'static': [], 'components': [], 'css': [], 'needs_build': None}
    if not task_directory_path:
        task_directory_path = os.path.join(self.opt['parlai_home'], 'parlai', 'mturk', 'tasks', self.opt['task'])
    self.task_files_to_copy['static'].append(os.path.join(task_directory_path, 'frontend', 'static', 'cover_page.html'))
    try:
        frontend_contents = os.listdir(os.path.join(task_directory_path, 'frontend'))
        if 'package.json' in frontend_contents:
            self.task_files_to_copy['needs_build'] = os.path.join(task_directory_path, 'frontend')
        for dir in frontend_contents:
            if dir in self.task_files_to_copy:
                for file_name in os.listdir(os.path.join(task_directory_path, 'frontend', dir)):
                    self.task_files_to_copy[dir].append(os.path.join(task_directory_path, 'frontend', dir, file_name))
    except FileNotFoundError:
        pass