import json
import logging
import os
import pickle
import sqlite3
import time
import threading
import parlai.mturk.core.dev.shared_utils as shared_utils
from parlai.mturk.core.dev.agents import AssignState
@staticmethod
def get_full_conversation_data(task_group_id, conv_id, is_sandbox):
    """
        Gets all conversation data saved for a world.
        """
    target = 'sandbox' if is_sandbox else 'live'
    return_data = {'custom_data': {}, 'worker_data': {}}
    target_dir = os.path.join(data_dir, target, task_group_id, conv_id)
    target_dir_custom = os.path.join(target_dir, 'custom')
    custom_file = os.path.join(target_dir_custom, 'data.json')
    if os.path.exists(custom_file):
        custom_data = {}
        with open(custom_file, 'r') as infile:
            custom_data = json.load(infile)
        pickle_file = os.path.join(target_dir_custom, 'data.pickle')
        if os.path.exists(pickle_file):
            with open(pickle_file, 'rb') as infile:
                custom_data['needs-pickle'] = pickle.load(infile)
        return_data['custom_data'] = custom_data
    target_dir_workers = os.path.join(target_dir, 'workers')
    for w_file in os.listdir(target_dir_workers):
        w_id = w_file.split('.json')[0]
        worker_file = os.path.join(target_dir_workers, w_file)
        with open(worker_file, 'r') as infile:
            return_data['worker_data'][w_id] = json.load(infile)
    return return_data