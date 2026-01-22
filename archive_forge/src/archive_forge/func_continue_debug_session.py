import copy
import json
import logging
import os
import signal
import subprocess
import sys
import time
import traceback
import urllib
import urllib.parse
import warnings
import shutil
from datetime import datetime
from typing import Optional, Set, List, Tuple
import click
import psutil
import yaml
import ray
import ray._private.ray_constants as ray_constants
import ray._private.services as services
from ray._private.utils import (
from ray._private.internal_api import memory_summary
from ray._private.storage import _load_class
from ray._private.usage import usage_lib
from ray.autoscaler._private.cli_logger import add_click_logging_options, cf, cli_logger
from ray.autoscaler._private.commands import (
from ray.autoscaler._private.constants import RAY_PROCESSES
from ray.autoscaler._private.fake_multi_node.node_provider import FAKE_HEAD_NODE_ID
from ray.util.annotations import PublicAPI
def continue_debug_session(live_jobs: Set[str]):
    """Continue active debugging session.

    This function will connect 'ray debug' to the right debugger
    when a user is stepping between Ray tasks.
    """
    active_sessions = ray.experimental.internal_kv._internal_kv_list('RAY_PDB_', namespace=ray_constants.KV_NAMESPACE_PDB)
    for active_session in active_sessions:
        if active_session.startswith(b'RAY_PDB_CONTINUE'):
            data = ray.experimental.internal_kv._internal_kv_get(active_session, namespace=ray_constants.KV_NAMESPACE_PDB)
            if json.loads(data)['job_id'] not in live_jobs:
                ray.experimental.internal_kv._internal_kv_del(active_session, namespace=ray_constants.KV_NAMESPACE_PDB)
                continue
            print('Continuing pdb session in different process...')
            key = b'RAY_PDB_' + active_session[len('RAY_PDB_CONTINUE_'):]
            while True:
                data = ray.experimental.internal_kv._internal_kv_get(key, namespace=ray_constants.KV_NAMESPACE_PDB)
                if data:
                    session = json.loads(data)
                    if 'exit_debugger' in session or session['job_id'] not in live_jobs:
                        ray.experimental.internal_kv._internal_kv_del(key, namespace=ray_constants.KV_NAMESPACE_PDB)
                        return
                    host, port = session['pdb_address'].split(':')
                    ray.util.rpdb._connect_pdb_client(host, int(port))
                    ray.experimental.internal_kv._internal_kv_del(key, namespace=ray_constants.KV_NAMESPACE_PDB)
                    continue_debug_session(live_jobs)
                    return
                time.sleep(1.0)