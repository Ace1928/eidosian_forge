import asyncio
import inspect
import logging
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from os import path as osp
from jupyter_server.serverapp import aliases, flags
from jupyter_server.utils import pathname2url, urljoin
from tornado.ioloop import IOLoop
from tornado.iostream import StreamClosedError
from tornado.websocket import WebSocketClosedError
from traitlets import Bool, Unicode
from .labapp import LabApp, get_app_dir
from .tests.test_app import TestEnv
def run_browser_sync(url):
    """Run the browser test and return an exit code."""
    target = osp.join(get_app_dir(), 'browser_test')
    if not osp.exists(osp.join(target, 'node_modules')):
        os.makedirs(target)
        subprocess.call(['npm', 'init', '-y'], cwd=target)
        subprocess.call(['npm', 'install', 'playwright@^1.9.2'], cwd=target)
    subprocess.call(['npx', 'playwright', 'install'], cwd=target)
    shutil.copy(osp.join(here, 'browser-test.js'), osp.join(target, 'browser-test.js'))
    return subprocess.check_call(['node', 'browser-test.js', url], cwd=target)