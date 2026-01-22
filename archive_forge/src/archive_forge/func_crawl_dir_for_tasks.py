from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import importlib
import inspect
import json
import logging
import os
import time
import threading
import traceback
import asyncio
import sh
import shlex
import shutil
import subprocess
import uuid
import tornado.ioloop
import tornado.web
import tornado.websocket
import tornado.escape
from parlai.mturk.core.mturk_data_handler import MTurkDataHandler
from parlai.mturk.core.mturk_manager import MTurkManager
from parlai.mturk.webapp.run_mocks.mock_turk_manager import MockTurkManager
from typing import Dict, Any
from parlai import __path__ as parlai_path  # type: ignore
def crawl_dir_for_tasks(search_dir):
    found_dirs = {}
    contents = os.listdir(search_dir)
    for sub_dir in contents:
        full_sub_dir = os.path.join(search_dir, sub_dir)
        if os.path.exists(os.path.join(full_sub_dir, 'run.py')):
            found_dirs[sub_dir] = full_sub_dir
        if os.path.isdir(full_sub_dir):
            found_dirs.update(crawl_dir_for_tasks(full_sub_dir))
    return found_dirs