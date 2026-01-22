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
def get_conversation_data(task_group_id, conv_id, worker_id, is_sandbox):
    """
        A poorly named function that gets conversation data for a worker.
        """
    result = {'had_data_dir': False, 'had_run_dir': False, 'had_conversation_dir': False, 'had_worker_dir': False, 'had_worker_file': False, 'data': None}
    target = 'sandbox' if is_sandbox else 'live'
    search_dir = os.path.join(data_dir, target)
    if not os.path.exists(search_dir):
        return result
    result['had_data_dir'] = True
    search_dir = os.path.join(search_dir, task_group_id)
    if not os.path.exists(search_dir):
        return result
    result['had_run_dir'] = True
    search_dir = os.path.join(search_dir, conv_id)
    if not os.path.exists(search_dir):
        return result
    result['had_conversation_dir'] = True
    search_dir = os.path.join(search_dir, 'workers')
    if not os.path.exists(search_dir):
        return result
    result['had_worker_dir'] = True
    target_filename = os.path.join(search_dir, '{}.json'.format(worker_id))
    if not os.path.exists(target_filename):
        return result
    result['had_worker_file'] = True
    with open(target_filename, 'r') as target_file:
        result['data'] = json.load(target_file)
    return result