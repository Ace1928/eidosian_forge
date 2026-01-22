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
def rebuild_source():
    if os.path.exists('dev/task_components'):
        print('removing old build')
        sh.rm(shlex.split('-rf {}'.format('dev/task_components')))
    os.mkdir('dev/task_components')
    copy_dirs = {}
    parlai_task_dir = os.path.join(parlai_path, 'mturk', 'tasks')
    tasks.update(crawl_dir_for_tasks(parlai_task_dir))
    if parlai_int_path is not None:
        parlai_internal_task_dir = os.path.join(parlai_int_path, 'mturk', 'tasks')
        tasks.update(crawl_dir_for_tasks(parlai_internal_task_dir))
    for task, task_dir in tasks.items():
        if os.path.exists(os.path.join(task_dir, 'frontend')):
            copy_dirs[task] = os.path.join(task_dir, 'frontend')
    for task, task_dir in copy_dirs.items():
        was_built = False
        if 'package.json' in os.listdir(task_dir):
            os.chdir(task_dir)
            packages_installed = subprocess.call(['npm', 'install'])
            if packages_installed != 0:
                raise Exception('please make sure npm is installed, otherwise view the above error for more info.')
            webpack_complete = subprocess.call(['npm', 'run', 'dev'])
            if webpack_complete != 0:
                raise Exception('Webpack appears to have failed to build your frontend. See the above error for more information.')
            was_built = True
        os.chdir(here)
        output_dir = os.path.join('dev/task_components', task)
        shutil.copytree(task_dir, output_dir)
        if was_built:
            shutil.copy2(os.path.join(output_dir, 'dist', 'custom.jsx'), os.path.join(output_dir, 'components', 'custom.jsx'))
        shutil.copy2(os.path.join(parlai_path, 'mturk', 'core', 'react_server', 'dev', 'components', 'core_components.jsx'), os.path.join(output_dir, 'components'))
    shutil.copy2(os.path.join(parlai_path, 'mturk', 'core', 'react_server', 'dev', 'components', 'core_components.jsx'), 'dev/task_components')
    packages_installed = subprocess.call(['npm', 'install'])
    if packages_installed != 0:
        raise Exception('please make sure npm is installed, otherwise view the above error for more info.')
    webpack_complete = subprocess.call(['npm', 'run', 'dev'])
    if webpack_complete != 0:
        raise Exception('Webpack appears to have failed to build your frontend. See the above error for more information.')